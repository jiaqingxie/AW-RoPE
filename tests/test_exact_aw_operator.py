from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from exact_aw_operator import exact_walk_resolvent, install_exact_backend
from exact_aw_diagnostics import production, evaluate

AW=production()


def fixture():
    torch.manual_seed(41)
    x=torch.randn(5,6,dtype=torch.double,requires_grad=True)
    e=torch.tensor([[0,1,1,2,2,0,0],[1,0,2,1,0,2,1]])
    a=torch.randn(7,dtype=torch.double,requires_grad=True)
    f=torch.randn(3,dtype=torch.double,requires_grad=True)
    w=torch.rand(7,dtype=torch.double,requires_grad=True)
    z=torch.tensor(.6,dtype=torch.double,requires_grad=True)
    return x,e,a,f,w,z


def test_exact_matches_long_production_recurrence_and_all_gradients():
    x,e,a,f,w,z=fixture()
    exact=exact_walk_resolvent(x,e,a,f,z=z,edge_weight=w)
    finite=AW.truncated_walk_resolvent(x,e,a,f,z=z,num_steps=90,edge_weight=w)
    torch.testing.assert_close(exact,finite,atol=1e-12,rtol=1e-12)
    for left,right in zip(torch.autograd.grad(exact.square().sum(),(x,a,f,w,z)),
                          torch.autograd.grad(finite.square().sum(),(x,a,f,w,z))):
        torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)


def test_numerical_gradcheck_including_z_and_weights():
    x,e,a,f,w,z=fixture()
    assert torch.autograd.gradcheck(lambda x,a,f,w,z: exact_walk_resolvent(x,e,a,f,z=z,edge_weight=w),
                                   (x,a,f,w,z),atol=1e-5,rtol=1e-4,fast_mode=True)


def test_batching_isolation_padding_and_permutation():
    x,e,a,f,w,z=fixture()
    single=exact_walk_resolvent(x,e,a,f,z=z,edge_weight=w)
    x2=torch.cat((x,x[:3]))
    both=exact_walk_resolvent(x2,torch.cat((e,e+5),1),a.repeat(2),f,z=z,
        edge_weight=w.repeat(2),batch=torch.tensor([0]*5+[1]*3))
    torch.testing.assert_close(single,both[:5])
    torch.testing.assert_close(single[:3],both[5:])
    permutation=torch.tensor([3,0,4,1,2]); inverse=torch.argsort(permutation)
    shuffled=exact_walk_resolvent(x[permutation],inverse[e],a,f,z=z,edge_weight=w)
    torch.testing.assert_close(shuffled,single[permutation])


@pytest.mark.parametrize("z",[0.,.5,.99])
def test_empty_edges_identity(z):
    x=torch.randn(3,4,dtype=torch.double)
    result=exact_walk_resolvent(x,torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.ones(2),z=z)
    torch.testing.assert_close(x,result,atol=0,rtol=0)


@pytest.mark.parametrize("z",[-.1,1.,float("nan")])
def test_invalid_z(z):
    x,e,a,f,w,_=fixture()
    with pytest.raises(ValueError): exact_walk_resolvent(x,e,a,f,z=z)


def test_reject_cross_graph_edges():
    x,e,a,f,w,z=fixture()
    with pytest.raises(ValueError,match="cross-graph"):
        exact_walk_resolvent(x,e,a,f,z=z,batch=torch.tensor([0,0,1,1,1]))


def test_curve_bounds_and_convergence():
    result=evaluate()
    assert result["valid"] and len(result["rows"])==32
    for z in (.4,.6,.8,.95):
        rows=[r for r in result["rows"] if r["z"]==z]
        assert rows[-1]["forward_relative"] < rows[0]["forward_relative"]
        assert rows[-1]["gradient_relative_l2"] < rows[0]["gradient_relative_l2"]


def test_full_graph_layer_state_and_backward():
    pytest.importorskip("local_attention")
    from torch_geometric.data import Batch,Data
    layer=ROOT/"external/Graph-RoPE/graphgps/layer"
    for name,path in (("exact_test_graphgps",layer.parent),("exact_test_graphgps.layer",layer)):
        package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
    modules=[]
    for name in ("aw_rope","graphrope"):
        spec=importlib.util.spec_from_file_location("exact_test_graphgps.layer."+name,layer/(name+".py"))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);modules.append(module)
    aw,graph=modules
    config=types.SimpleNamespace(num_steps=90,z=.6,field_hidden_dim=1,learnable_z=True)
    torch.manual_seed(8);finite=graph.GraphRoPE(0,8,2,dropout=0.,positional_method="aw",aw_cfg=config).double()
    torch.manual_seed(8);exact=graph.GraphRoPE(0,8,2,dropout=0.,positional_method="aw",aw_cfg=config).double()
    assert all(torch.equal(v,exact.state_dict()[k]) for k,v in finite.state_dict().items())
    data=Data(x=torch.randn(5,8,dtype=torch.double),edge_index=fixture()[1])
    batch=Batch.from_data_list([data,data.clone()])
    left=finite(batch)
    original=install_exact_backend(aw.AnalyticWalkRoPE,graph.GraphRoPE)
    try:
        right=exact(batch)
        torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-9)
        right.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in exact.parameters())
        assert exact.aw_rope._exact_aw_batch is None
    finally:
        aw.AnalyticWalkRoPE._transport_single,graph.GraphRoPE.forward=original
