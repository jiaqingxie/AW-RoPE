import math
import torch
from context_route_transport import generate, power, Predictor, evaluate


def test_production_transport_matches_two_route_formula():
    torch.manual_seed(392)
    d = generate(4, 915)
    a = torch.randn(len(d['x']), 4, dtype=torch.float64)
    oriented = torch.stack([a, -a], -1).flatten(1)
    delta = a[:, 0]+a[:, 1]-a[:, 2]-a[:, 3]
    torch.testing.assert_close(power(oriented, d['probe']), (delta/2).cos().square(), atol=1e-12, rtol=1e-12)
    separate = torch.cat([power(oriented[i:i+1], d['probe'][i:i+1]) for i in range(len(a))])
    torch.testing.assert_close(power(oriented, d['probe']), separate)


def test_context_field_witness_and_zero_intervention():
    d = generate(8, 912)
    upper, lower = d['x'][:, 1, 4], d['x'][:, 2, 4]
    a = math.pi/4*torch.stack([upper, upper, lower, lower], -1)
    a = torch.stack([a, -a], -1).flatten(1)
    torch.testing.assert_close(power(a, d['probe']), d['y'], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(power(torch.zeros_like(a), d['probe']), torch.ones_like(d['y']))


def test_static_and_gradient_obstructions_and_shared_initialization():
    d = generate(8, 910)
    for seed in range(5):
        gradient = Predictor('gradient', seed)
        aw = Predictor('unrestricted', seed)
        for key, value in gradient.state_dict().items():
            assert torch.equal(value, aw.state_dict()[key])
        a = gradient.displacement(d['x'])
        torch.testing.assert_close(a[:, 0]+a[:, 2]-a[:, 4]-a[:, 6], torch.zeros(len(a), dtype=a.dtype), atol=1e-14, rtol=0)
        torch.testing.assert_close(gradient(d['x'], d['probe']), torch.ones_like(d['y']))
        assert evaluate(gradient, d)['accuracy'] == .5
        static = Predictor('static', seed)
        p = static(d['x'], d['probe']).reshape(-1, 4)
        torch.testing.assert_close(p, p[:, :1].expand_as(p))


def test_balanced_contexts_and_disjoint_nuisance_groups():
    signatures = []
    for seed in (26091400, 26091401, 26091402):
        d = generate(16, seed)
        x = d['x'].reshape(-1, 4, 4, 7)
        assert torch.equal(x[:, :, :, 5:], x[:, :1, :, 5:].expand_as(x[:, :, :, 5:]))
        assert d['y'].reshape(-1, 4).sum(1).eq(2).all()
        signatures.append({tuple(row.tolist()) for row in x[:, 0, :, 5:].flatten(1)})
    assert all(not signatures[i] & signatures[j] for i in range(3) for j in range(i))
