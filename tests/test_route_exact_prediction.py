import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import route_phase_exact_prediction as e


@pytest.mark.parametrize('readout',['endpoint','attention'])
@pytest.mark.parametrize('zero',[False,True])
def test_exact_matches_converged_sparse_outputs_and_input_vjp(readout,zero):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    exact=e.ExactRoutePredictor(readout=readout,L=8,z=.8,seed=43).double()
    torch.manual_seed(43)
    sparse=e.SparseRoutePredictor('aw',readout=readout,L=160,z=.8,seed=43).double()
    sparse.load_state_dict(exact.state_dict())
    # The legacy zero-phase helper reads a Python float; Exact reads the
    # production float32-initialized z buffer. Match its value for float64 QA.
    if zero: sparse.z=float(exact.pos[0].z)
    assert exact.parameter_audit==sparse.parameter_audit
    data=e.base.generate(2,478,motifs=2,route_length=8)
    data={k:v.double() if v.is_floating_point() else v for k,v in data.items() if isinstance(v,torch.Tensor)}
    data['x'].requires_grad_(True)
    mode='zero-phase' if zero else None
    a=exact(data,intervention=mode);b=sparse(data,intervention=mode)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    ga=torch.autograd.grad(a.sum(),data['x'],retain_graph=True)[0]
    gb=torch.autograd.grad(b.sum(),data['x'])[0]
    torch.testing.assert_close(ga,gb,atol=2e-9,rtol=2e-9)
    if readout=='endpoint' and zero:
        torch.testing.assert_close(a[::2],a[1::2],atol=1e-12,rtol=0)


def test_exact_batching_is_independent():
    torch.set_num_threads(1);torch.manual_seed(73)
    model=e.ExactRoutePredictor(readout='attention',L=8,z=.8,seed=73).double()
    data=e.base.generate(2,567,motifs=2,route_length=8)
    data={k:v.double() if v.is_floating_point() else v for k,v in data.items() if isinstance(v,torch.Tensor)}
    with torch.no_grad():
        full=model(data)
        single=torch.cat([model(e.base.batch(data,slice(i,i+1))) for i in range(4)])
    torch.testing.assert_close(full,single,atol=1e-12,rtol=1e-12)
