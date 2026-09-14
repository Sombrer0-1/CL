"""Avalanche 0.6.0 GSS compatibility plus dynamic mem_size (A01, A10)."""

from __future__ import annotations

import torch
from avalanche.training.plugins.gss_greedy import GSS_greedyPlugin


def patch_gss_plugin() -> None:
    """Keep official GSS_greedy() constructible on this PyTorch build."""
    if getattr(GSS_greedyPlugin, "_orion_mem_strength_patch", False):
        return
    original_init = GSS_greedyPlugin.__init__

    def patched_init(self, mem_size=200, mem_strength=5, input_size=None):
        if input_size is None:
            input_size = []
        original_init(
            self,
            mem_size=mem_size,
            mem_strength=int(mem_strength),
            input_size=input_size,
        )

    GSS_greedyPlugin.__init__ = patched_init  # type: ignore[method-assign]
    GSS_greedyPlugin._orion_mem_strength_patch = True


class AdaptiveGSSPlugin(GSS_greedyPlugin):
    """Official GSS selection with occupancy resize that does not restore discards."""

    def __init__(self, mem_size=200, mem_strength=5, input_size=None):
        if input_size is None:
            input_size = []
        super().__init__(
            mem_size=int(mem_size),
            mem_strength=int(mem_strength),
            input_size=list(input_size),
        )
        self.auxiliary_visits = 0
        self.selection_calls = 0
        self.full_buffer_score_calls = 0

    def before_training_exp(self, strategy, num_workers=0, shuffle=True, **kwargs):
        self.auxiliary_visits = 0
        self.selection_calls = 0
        self.full_buffer_score_calls = 0
        return super().before_training_exp(
            strategy, num_workers=num_workers, shuffle=shuffle, **kwargs
        )

    def get_batch_sim(self, strategy, grad_dims, batch_x, batch_y):
        self.full_buffer_score_calls += 1
        mem_grads = self.get_rand_mem_grads(strategy, grad_dims, len(batch_x))
        if mem_grads.numel() == 0:
            return torch.zeros(1, device=self.device), mem_grads
        strategy.model.zero_grad()
        loss = strategy._criterion(strategy.model.forward(batch_x), batch_y)
        loss.backward()
        batch_grad = self.get_grad_vector(strategy.model.parameters, grad_dims).unsqueeze(0)
        sims = self.cosine_similarity(mem_grads, batch_grad)
        if sims.numel() == 0:
            return torch.zeros(1, device=self.device), mem_grads
        return torch.max(sims), mem_grads

    def get_each_batch_sample_sim(self, strategy, grad_dims, mem_grads, batch_x, batch_y):
        self.auxiliary_visits += int(batch_x.size(0))
        if mem_grads is None or mem_grads.numel() == 0:
            return torch.zeros(batch_x.size(0), device=strategy.device) + 0.1
        cosine_sim = torch.zeros(batch_x.size(0), device=strategy.device)
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            strategy.model.zero_grad()
            ptloss = strategy._criterion(
                strategy.model.forward(x.unsqueeze(0)), y.unsqueeze(0)
            )
            ptloss.backward()
            this_grad = self.get_grad_vector(strategy.model.parameters, grad_dims).unsqueeze(0)
            sims = self.cosine_similarity(mem_grads, this_grad)
            cosine_sim[i] = torch.max(sims) if sims.numel() else 0.1
        return cosine_sim

    def get_rand_mem_grads(self, strategy, grad_dims, gss_batch_size):
        occupied = int(self.ext_mem_list_current_index)
        if occupied <= 0 or gss_batch_size <= 0:
            return torch.zeros(0, sum(grad_dims), dtype=torch.float32, device=self.device)
        temp_gss_batch_size = min(int(gss_batch_size), occupied)
        official_subs = occupied // max(1, int(gss_batch_size))
        # Official uses occupied // batch, which is 0 when mem_size < batch
        # (MAX-P). Use one subset of the occupied buffer in that case.
        num_mem_subs = min(self.mem_strength, official_subs if official_subs >= 1 else 1)
        self.auxiliary_visits += int(num_mem_subs * temp_gss_batch_size)
        mem_grads = torch.zeros(
            num_mem_subs,
            sum(grad_dims),
            dtype=torch.float32,
            device=self.device,
        )
        shuffled = torch.randperm(occupied, device=self.device)
        for i in range(num_mem_subs):
            start = i * temp_gss_batch_size
            random_batch_inds = shuffled[start : start + temp_gss_batch_size]
            batch_x = self.ext_mem_list_x[random_batch_inds].to(strategy.device)
            batch_y = self.ext_mem_list_y[random_batch_inds].to(strategy.device)
            strategy.model.zero_grad()
            loss = strategy._criterion(strategy.model.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(self.get_grad_vector(strategy.model.parameters, grad_dims))
        return mem_grads

    def after_forward(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """Official GSS insert/replace, plus MAX-P guards (A23).

        Official code assumes occupancy >= batch and non-equal buffer scores.
        """
        self.selection_calls += 1
        strategy.model.eval()
        grad_dims = [param.data.numel() for param in strategy.model.parameters()]
        occupied = int(self.ext_mem_list_current_index)
        place_left = int(self.ext_mem_list_x.size(0) - occupied)
        mb_x = strategy.mb_x
        mb_y = strategy.mb_y
        if place_left <= 0:
            if occupied <= 0:
                strategy.model.train()
                return
            n = min(int(mb_x.size(0)), occupied)
            batch_x = mb_x[:n]
            batch_y = mb_y[:n]
            batch_sim, mem_grads = self.get_batch_sim(
                strategy, grad_dims, batch_x=batch_x, batch_y=batch_y
            )
            sim_value = (
                float(batch_sim.reshape(-1)[0].item())
                if torch.is_tensor(batch_sim)
                else float(batch_sim)
            )
            if sim_value < 0:
                buffer_score = self.buffer_score[:occupied].detach().cpu()
                span = (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                buffer_sim = (buffer_score - torch.min(buffer_score)) / span
                if float(buffer_sim.sum()) <= 0:
                    buffer_sim = torch.ones_like(buffer_sim)
                index = torch.multinomial(buffer_sim, n, replacement=False).to(strategy.device)
                batch_item_sim = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, batch_x, batch_y
                )
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(1)
                pair = torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1)
                pair = pair.clamp_min(1e-8)
                outcome = torch.multinomial(pair, 1, replacement=False)
                added_indx = torch.arange(end=batch_item_sim.size(0), device=strategy.device)
                sub_index = outcome.squeeze(1).bool()
                self.ext_mem_list_x[index[sub_index]] = batch_x[added_indx[sub_index]].clone()
                self.ext_mem_list_y[index[sub_index]] = batch_y[added_indx[sub_index]].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
        else:
            offset = min(place_left, int(mb_x.size(0)))
            updated_mb_x = mb_x[:offset]
            updated_mb_y = mb_y[:offset]
            if occupied == 0:
                batch_sample_memory_cos = torch.zeros(
                    updated_mb_x.size(0), device=strategy.device
                ) + 0.1
            else:
                mem_grads = self.get_rand_mem_grads(
                    strategy=strategy,
                    grad_dims=grad_dims,
                    gss_batch_size=len(mb_x),
                )
                batch_sample_memory_cos = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, updated_mb_x, updated_mb_y
                )
            curr_idx = self.ext_mem_list_current_index
            self.ext_mem_list_x[curr_idx : curr_idx + offset].data.copy_(updated_mb_x)
            self.ext_mem_list_y[curr_idx : curr_idx + offset].data.copy_(updated_mb_y)
            self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                batch_sample_memory_cos
            )
            self.ext_mem_list_current_index += offset
        strategy.model.train()

    def resize(self, mem_size: int) -> None:
        new_size = int(mem_size)
        if new_size < 0:
            raise ValueError("GSS mem_size must be non-negative")
        old_x = self.ext_mem_list_x
        old_y = self.ext_mem_list_y
        old_score = self.buffer_score
        keep = min(int(self.ext_mem_list_current_index), new_size)
        device = old_x.device
        feat_shape = tuple(old_x.shape[1:])
        if new_size == 0:
            self.ext_mem_list_x = torch.zeros(0, *feat_shape, device=device, dtype=old_x.dtype)
            self.ext_mem_list_y = torch.zeros(0, device=device, dtype=old_y.dtype)
            self.buffer_score = torch.zeros(0, device=device, dtype=old_score.dtype)
            self.ext_mem_list_current_index = 0
            self.mem_size = 0
            return
        new_x = old_x.new_zeros((new_size, *feat_shape))
        new_y = old_y.new_zeros((new_size,))
        new_score = old_score.new_zeros((new_size,))
        if keep > 0:
            new_x[:keep].copy_(old_x[:keep])
            new_y[:keep].copy_(old_y[:keep])
            new_score[:keep].copy_(old_score[:keep])
        self.ext_mem_list_x = new_x
        self.ext_mem_list_y = new_y
        self.buffer_score = new_score
        self.ext_mem_list_current_index = keep
        self.mem_size = new_size
