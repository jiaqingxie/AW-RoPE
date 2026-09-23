import torch
import full_network_field_prediction as f


def signature(pattern):
    n=len(pattern);result=[]
    for i,c in enumerate(pattern):
        rows=[c]
        for hop in (1,2):
            rows += [int(pattern[(i-hop)%n]==k)+int(pattern[(i+hop)%n]==k) for k in range(3)]
        result.append(tuple(rows))
    return sorted(result)


def test_exact_integer_collision_and_nonisomorphic_coloring():
    a,b=f.PATTERNS
    assert signature(a)==signature(b)
    assert all(a!=b[k:]+b[:k] and a!=b[::-1][k:]+b[::-1][:k] for k in range(len(a)))


def test_full_attention_zero_local_gradient_collision_and_aw_separation():
    torch.set_num_threads(1)
    data=f.generate(5,19813)
    for seed in (17,31):
        for arm in ('zero','static','gradient'):
            torch.manual_seed(seed);model=f.Model(arm,seed=seed)
            logits=model(data)
            torch.testing.assert_close(logits[::2],logits[1::2],atol=1e-11,rtol=0)
    torch.manual_seed(17);aw=f.Model('unrestricted',seed=17)
    logits=aw(data)
    assert float((logits[::2]-logits[1::2]).abs().max()) > 1e-7
    logits=aw(data,intervention='zero-phase')
    torch.testing.assert_close(logits[::2],logits[1::2],atol=1e-11,rtol=0)


def test_data_pairing_permutation_equivariance_and_parameter_budgets():
    d=f.generate(3,7391)
    initial={}
    for arm in f.ARMS:
        torch.manual_seed(17);m=f.Model(arm,seed=17)
        initial[arm]=m.audit['total_parameters']
        permutation=torch.randperm(12);inverse=torch.argsort(permutation)
        changed={**d,'x':d['x'][:,permutation], 't':d['t'][:,permutation], 'edges':inverse[d['edges']]}
        torch.testing.assert_close(m(d),m(changed),atol=1e-11,rtol=1e-11)
    assert len(set(initial.values()))==1,initial


def test_projected_flat_operator_really_has_zero_cycle_sum():
    torch.set_num_threads(1)
    d=f.generate(2,930);torch.manual_seed(17)
    model=f.Model('projected-flat',seed=17)
    pos=model.pos[0]
    original=pos._transport_single
    seen=[]
    # Capture the displacement actually consumed by transport, after projection.
    import inspect
    signature=inspect.signature(original)
    def capture(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        a=bound.arguments['displacement'].reshape(4,24)
        seen.append(a[:,::2].sum(-1))
        return original(*args,**kwargs)
    pos._transport_single=capture
    model(d)
    assert seen
    for circulation in seen:
        torch.testing.assert_close(circulation,torch.zeros_like(circulation),atol=1e-11,rtol=0)
