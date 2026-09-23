import torch

from aw_rope import AntisymmetricEdgeField


def test_learned_edge_field_is_antisymmetric() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]])
    edge_attr = torch.randn(4, 2)
    reverse_attr = edge_attr[torch.tensor([1, 0, 3, 2])]
    field = AntisymmetricEdgeField(5, edge_dim=2, hidden_dim=12)
    displacement = field(x, edge_index, edge_attr, reverse_attr)
    torch.testing.assert_close(displacement, -displacement[torch.tensor([1, 0, 3, 2])])


def test_field_backpropagates() -> None:
    x = torch.randn(3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    field = AntisymmetricEdgeField(4, hidden_dim=8, max_displacement=2.0)
    displacement = field(x, edge_index)
    displacement[0].backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert displacement.abs().max() <= 2.0

