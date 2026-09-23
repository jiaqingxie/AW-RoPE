"""Independent numerical check of the new directional derivative tail bound."""
import pytest
import torch


@pytest.mark.parametrize("steps", [0,1,4,8,16])
def test_trainable_decay_and_phase_directional_bound(steps):
    dtype=torch.float64
    p=torch.tensor([[0.,.7,.3],[.2,0.,.8],[.6,.4,0.]],dtype=dtype)
    phase=torch.tensor([[0.,.4,-.7],[-.4,0.,.9],[.7,-.9,0.]],dtype=dtype)
    theta=torch.tensor(.3,dtype=dtype,requires_grad=True)
    def matrix(t):
        return torch.sigmoid(t)*p*torch.exp(1j*t*phase)
    b=matrix(theta)
    def error(t):
        b=matrix(t); eye=torch.eye(3,dtype=torch.complex128)
        finite=eye;power=eye
        for _ in range(steps): power=power@b;finite=finite+power
        return torch.view_as_real(torch.linalg.solve(eye-b,eye)-finite)
    derivative=torch.view_as_complex(torch.autograd.functional.jacobian(error,theta).contiguous())
    db=torch.view_as_complex(torch.autograd.functional.jacobian(lambda t:torch.view_as_real(matrix(t)),theta).contiguous())
    r=float(torch.linalg.matrix_norm(b,ord=float("inf")))
    g=float(torch.linalg.matrix_norm(db,ord=float("inf")))
    bound=g*r**steps*((steps+1)-steps*r)/(1-r)**2
    assert float(torch.linalg.matrix_norm(derivative,ord=float("inf"))) <= bound+1e-12
