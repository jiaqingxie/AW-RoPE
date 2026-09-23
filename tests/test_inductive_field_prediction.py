import torch
import inductive_field_prediction as f


def signature(pattern):
    n=len(pattern)
    return sorted((c,*[int(pattern[(i-h)%n]==k)+int(pattern[(i+h)%n]==k) for h in (1,2) for k in range(3)]) for i,c in enumerate(pattern))


def test_infinite_family_padding_certificate_and_nonisomorphic_labels():
    # Analytically: padding a common length-3 zero run replaces one signature
    # by two identical boundary signatures and k-1 interior zero signatures.
    from collections import Counter
    a0,b0=f.patterns(12)
    baseline=Counter(signature(a0));assert baseline==Counter(signature(b0))
    delta1=Counter(signature(f.patterns(13)[0]));delta1.subtract(baseline)
    pure=(0,2,0,0,2,0,0)
    for n in range(12,97):
        a,b=f.patterns(n)
        assert signature(a)==signature(b)
        assert all(a!=b[i:]+b[:i] and a!=b[::-1][i:]+b[::-1][:i] for i in range(n))
        if n>12:
            delta=Counter(signature(a));delta.subtract(baseline)
            expected=delta1.copy();expected[pure]+=n-13
            assert +delta==+expected and -delta==-expected


def test_no_persistent_ids_disjoint_sizes_and_paired_randomness():
    sets=[set(x) for x in [f.TRAIN_SIZES,f.VAL_SIZES,f.TEST_SIZES]]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    d=f.generate(3,15900,19)
    assert 'template_id' not in d
    assert torch.equal(d['random_a'][::2],d['random_a'][1::2])
    assert torch.equal(d['t'][::2],d['t'][1::2])
    assert torch.equal(d['edges'][::2],d['edges'][1::2])


def test_complete_network_collision_even_for_learned_static_matrices():
    torch.set_num_threads(1)
    for n in (12,15,24,32):
        d=f.generate(2,89411+n,n)
        for arm in ('zero','gradient','learned-static'):
            torch.manual_seed(17);m=f.Model(arm,17)
            if arm=='learned-static':
                with torch.no_grad():m.pos[0].matrices.normal_()
            logits=m(d)
            torch.testing.assert_close(logits[::2],logits[1::2],atol=1e-10,rtol=0)


def test_constant_pi_connection_also_collides():
    d=f.generate(2,481,15);torch.manual_seed(17);m=f.Model('learned-static',17)
    with torch.no_grad():
        for i in range(2):
            for hop in range(3):m.pos[0].matrices[i,hop].copy_(torch.eye(16)*(-.8)**hop)
    logits=m(d)
    torch.testing.assert_close(logits[::2],logits[1::2],atol=1e-10,rtol=0)


def test_permutation_covariance_and_equal_parameter_budgets():
    torch.set_num_threads(1)
    d=f.generate(2,1915,19);counts=[]
    permutation=torch.randperm(19);inverse=torch.argsort(permutation)
    changed={**d,'x':d['x'][:,permutation],'t':d['t'][:,permutation],'edges':inverse[d['edges']]}
    for arm in f.ARMS:
        torch.manual_seed(17);m=f.Model(arm,17);counts.append(m.audit['total_parameters'])
        torch.testing.assert_close(m(d),m(changed),atol=1e-10,rtol=1e-10)
    assert set(counts)=={4810}


def test_predictions_do_not_use_pair_order_or_other_graphs():
    torch.set_num_threads(1)
    d=f.generate(3,51813,19)
    order=torch.tensor([5,0,3,1,4,2])
    for arm in f.ARMS:
        torch.manual_seed(19);m=f.Model(arm,19)
        original=m(d)
        torch.testing.assert_close(m(f.full.batch(d,order)),original[order],atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(m(f.full.batch(d,slice(1,2))),original[1:2],atol=1e-10,rtol=1e-10)
