"""Process-local PyTorch allocator quota; excludes driver and non-Torch allocations.

On Jetson unified-memory hosts, ``get_device_properties().total_memory`` is the
shared pool (nvidia-smi discrete VRAM is N/A). The fraction still only binds
this process's PyTorch CUDA allocator; it is not a cgroup or board-level hard
limit, and must not be described as one.
"""

def install_device_quota(budget, device):
    if budget.get('enforcement') != 'device_allocator_enforced':
        return None
    import torch
    limit = budget.get('limit_bytes')
    if device.type != 'cuda' or budget.get('controlled_resource') != 'device':
        raise ValueError('allocator quota requires CUDA and controlled_resource=device')
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= total:
        raise ValueError('allocator limit_bytes must be a positive integer <= device total')
    torch.cuda.set_per_process_memory_fraction(limit / total, index)
    return {'mechanism':'torch_cuda_allocator_fraction','limit_bytes':limit,
            'fraction':limit/total,'total_device_bytes':total,
            'scope':'process PyTorch CUDA allocator; excludes driver/non-Torch allocations'}
