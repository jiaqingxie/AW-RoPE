"""Run a small differentiable AW-RoPE transport on CPU."""
import torch

from aw_rope import AWRoPE, edge_displacement_from_positions


def main():
    torch.manual_seed(0)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    positions = torch.arange(3, dtype=torch.float32)
    displacement = edge_displacement_from_positions(positions, edges)
    x = torch.randn(3, 64, requires_grad=True)
    rope = AWRoPE(dim=64, num_steps=16, initial_z=0.8, learnable_z=True)
    y = rope(x, edges, displacement)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"AW-RoPE output: {tuple(y.shape)}; forward and backward passed")


if __name__ == '__main__':
    main()
