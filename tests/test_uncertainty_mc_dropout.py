import torch
from torch import nn

from growth_twin.relief.elevation import enable_dropout, mc_dropout_predict


def _head() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Dropout(0.5), nn.Linear(16, 1))


def test_mc_dropout_varies_when_active() -> None:
    head = _head().eval()
    enable_dropout(head)
    x = torch.randn(4, 8)
    mean, var = mc_dropout_predict(lambda: head(x).squeeze(-1), 16)
    assert float(var.mean()) > 0.0
    assert mean.shape == (4,)


def test_mc_dropout_constant_when_disabled() -> None:
    head = _head().eval()
    x = torch.randn(4, 8)
    _, var = mc_dropout_predict(lambda: head(x).squeeze(-1), 8)
    assert float(var.max()) < 1e-12
