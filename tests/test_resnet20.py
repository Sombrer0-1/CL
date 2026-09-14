import torch

from orion_repro.models.resnet20 import count_parameters, resnet20


def test_resnet20_param_count_and_shapes():
    model = resnet20(num_classes=10)
    n = count_parameters(model)
    assert 260_000 < n < 280_000
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_eval_does_not_update_bn_or_params():
    model = resnet20(num_classes=10)
    model.train()
    bn = model.bn1
    running_mean = bn.running_mean.clone()
    params = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(4, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    assert torch.allclose(bn.running_mean, running_mean)
    for a, b in zip(params, model.parameters()):
        assert torch.equal(a, b.detach())
