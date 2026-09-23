"""CUDA-path equivalence and solver dispatch regressions."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_aw_operator as operator
from exact_aw_operator import exact_walk_resolvent, solve_complex_system


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [3, 10, 25])
def test_real_block_matches_complex_forward_and_all_input_gradients(dtype, n):
    torch.manual_seed(92)
    x = torch.randn(n, 8, dtype=dtype, requires_grad=True)
    e = torch.stack((torch.arange(n).repeat(2),
                     torch.cat(((torch.arange(n)+1) % n, (torch.arange(n)+2) % n))))
    a = torch.randn(2*n, dtype=dtype, requires_grad=True)
    f = torch.randn(4, dtype=dtype, requires_grad=True)
    w = torch.rand(2*n, dtype=dtype, requires_grad=True)
    z = torch.tensor(.8, dtype=dtype, requires_grad=True)
    left = exact_walk_resolvent(x, e, a, f, z=z, edge_weight=w, solve_backend="native")
    right = exact_walk_resolvent(x, e, a, f, z=z, edge_weight=w, solve_backend="real-block")
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(left, right, atol=tol, rtol=tol)
    for l, r in zip(torch.autograd.grad(left.square().mean(), (x,a,f,w,z)),
                    torch.autograd.grad(right.square().mean(), (x,a,f,w,z))):
        torch.testing.assert_close(l, r, atol=tol, rtol=tol)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n", [10, 25])
def test_training_shaped_batch_and_backward(device, n):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires actual CUDA runtime")
    torch.manual_seed(29)
    # Actual training sizes: WS-SPD has 10 nodes; Monochromatic has 25.
    a = torch.randn(16,32,n,n, dtype=torch.complex64) / 200
    a = (a + torch.eye(n)).requires_grad_()
    b = torch.randn(16,32,n,1, dtype=torch.complex64, requires_grad=True)
    ref = solve_complex_system(a, b, backend="native")
    ref_grads = torch.autograd.grad(ref.abs().square().mean(), (a,b))
    aa = a.detach().to(device).requires_grad_()
    bb = b.detach().to(device).requires_grad_()
    result = solve_complex_system(aa, bb, backend="real-block")
    if device == "cuda":
        assert "cusolver" in str(torch.backends.cuda.preferred_linalg_library()).lower()
    actual_grads = torch.autograd.grad(result.abs().square().mean(), (aa,bb))
    torch.testing.assert_close(result.cpu(), ref, atol=3e-5, rtol=3e-5)
    for actual, expected in zip(actual_grads, ref_grads):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-7, rtol=3e-4)


def test_cuda_auto_dispatch_never_calls_native_complex_solve(monkeypatch):
    configurations = []
    monkeypatch.setattr(operator, "configure_cuda_solver", lambda: configurations.append("cusolver"))
    class FakeMatrix:
        is_cuda = True
        shape = (2, 2)
        real = torch.eye(2)
        imag = torch.zeros(2, 2)
    original = torch.linalg.solve
    calls = []
    def checked(a, b):
        calls.append(a.shape)
        assert not a.is_complex()
        return original(a,b)
    monkeypatch.setattr(torch.linalg, "solve", checked)
    result = solve_complex_system(FakeMatrix(), torch.ones(2,1,dtype=torch.complex64))
    assert calls == [torch.Size([4,4])]
    assert configurations == ["cusolver"]
    torch.testing.assert_close(result, torch.ones(2,1,dtype=torch.complex64))


def test_cuda_solver_preference_is_selected_and_rechecked(monkeypatch):
    state = ["Magma"]
    calls = []
    def preference(value=None):
        if value is not None:
            calls.append(value)
            state[0] = value
        return state[0]
    monkeypatch.setattr(torch.backends.cuda, "preferred_linalg_library", preference)
    assert operator.configure_cuda_solver() == "cusolver"
    assert operator.configure_cuda_solver() == "cusolver"
    assert calls == ["cusolver"]
    state[0] = "Default"
    operator.configure_cuda_solver()
    assert calls == ["cusolver", "cusolver"]


def test_cuda_solver_preference_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda, "preferred_linalg_library", lambda *args: "Magma")
    with pytest.raises(RuntimeError, match="requires the cuSOLVER"):
        operator.configure_cuda_solver()
