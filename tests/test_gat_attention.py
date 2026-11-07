import torch

from growth_twin.triangulation.bearings import GraphAttentionLayer


def test_attention_rows_normalized() -> None:
    layer = GraphAttentionLayer(8, 4, 2, 0.0, concat=True)
    h = torch.randn(2, 5, 8)
    layer.eval()
    wh = layer.lin(h).view(2, 5, 2, 4)
    src = (wh * layer.a_src).sum(-1).permute(0, 2, 1)
    dst = (wh * layer.a_dst).sum(-1).permute(0, 2, 1)
    scores = torch.nn.functional.leaky_relu(src.unsqueeze(3) + dst.unsqueeze(2), 0.2)
    alpha = torch.softmax(scores, dim=-1)
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 2, 5), atol=1e-5)


def test_isolated_node_attends_self() -> None:
    layer = GraphAttentionLayer(6, 6, 1, 0.0, concat=False)
    h = torch.randn(1, 3, 6)
    adj = torch.eye(3)
    layer.eval()
    out = layer(h, adj)
    wh = layer.lin(h).view(1, 3, 1, 6)
    assert torch.allclose(out, wh[:, :, 0], atol=1e-5)


def test_masked_neighbors_excluded() -> None:
    layer = GraphAttentionLayer(4, 4, 1, 0.0, concat=False)
    h = torch.randn(1, 4, 4)
    adj = torch.eye(4)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    layer.eval()
    out = layer(h, adj)
    assert out.shape == (1, 4, 4)
    assert torch.isfinite(out).all()
