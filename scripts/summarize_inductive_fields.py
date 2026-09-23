"""Replay the frozen inductive cohort, including every prespecified seed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    root=args.root.resolve();m=json.loads((root/'manifest.json').read_text())
    source=Path(m['source']);sys.path.insert(0,str(source/'scripts'))
    import torch
    import inductive_field_prediction as f
    torch.set_num_threads(1)
    for name,digest in m['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    rows=[];shared={};dynamic={};tensor_hashes={}
    for trial in m['trials']:
        rd=root/trial['id'];r=json.loads((rd/'result.json').read_text())
        audit=json.loads((rd/'audit.json').read_text())
        assert f.sha(rd/'best.pt')==r['checkpoint_sha256']
        trajectory=[json.loads(line) for line in (rd/'trajectory.jsonl').read_text().splitlines()]
        assert min(trajectory,key=lambda x:x['validation']['bce'])==r['best']
        torch.manual_seed(trial['seed']);model=f.Model(trial['arm'],trial['seed'],trial['gain'],trial['readout'])
        assert model.audit['initial_sha256']==audit['initial_sha256']
        shared.setdefault(trial['seed'],set()).add(audit['shared_initial_sha256'])
        if trial['arm'] in ('unrestricted','gradient','projected-flat'):
            dynamic.setdefault(trial['seed'],set()).add(audit['initial_sha256'])
        model.load_state_dict(torch.load(rd/'best.pt',weights_only=True))
        test={n:f.generate(128,trial['data_seed']+2000+n,n) for n in f.TEST_SIZES}
        assert {str(n):f.state_hash(d) for n,d in test.items()}==r['test_hashes']
        tensor_hashes.setdefault(trial['data_seed'],set()).add(json.dumps(r['test_hashes'],sort_keys=True))
        metrics=f.evaluate(model,test);assert metrics==r['test']
        val={n:f.generate(32,trial['data_seed']+1000+n,n) for n in f.VAL_SIZES}
        assert {str(n):f.state_hash(d) for n,d in val.items()}==audit['val_hashes']
        assert f.evaluate(model,val)==r['best']['validation']
        extra=f.evaluate(model,{n:f.generate(64,trial['data_seed']+3000+n,n) for n in (48,64)})
        assert extra==r['extrapolation']
        field_audit={}
        if trial['arm'] in ('unrestricted','gradient','projected-flat'):
            import inspect
            captured=[];original=model.pos[0]._transport_single;sig=inspect.signature(original)
            n=next(iter(test));d=f.full.batch(test[n],slice(0,64))
            def capture(*a,**kw):
                displacement=sig.bind(*a,**kw).arguments['displacement']
                captured.append(displacement.reshape(-1,2*n).detach())
                return original(*a,**kw)
            model.pos[0]._transport_single=capture
            with torch.no_grad():model(d)
            model.pos[0]._transport_single=original
            a=captured[0];cycle=a[:,::2].sum(-1)
            field_audit=dict(max_abs_cycle_sum=float(cycle.abs().max()),
                             max_paired_cycle_sum_difference=float((cycle[::2]-cycle[1::2]).abs().max()),
                             paired_edge_field_rms_difference=float((a[::2]-a[1::2]).square().mean().sqrt()))
            if trial['arm'] in ('gradient','projected-flat'):
                assert field_audit['max_abs_cycle_sum']<1e-9
        zero=None
        if trial['arm'] in ('unrestricted','gradient','projected-flat','random-static','wire','wire-mixing'):
            if trial['arm'] in ('wire','wire-mixing'):
                with torch.no_grad():
                    for pos in model.pos:pos.weight.zero_()
            zero=sum(f.full.evaluate(model,d,intervention='zero-phase')[0]['accuracy'] for d in test.values())/len(test)
        row=dict(arm=trial['arm'],seed=trial['seed'],accuracy=metrics['accuracy'],
                 validation_accuracy=r['best']['validation']['accuracy'],best_step=r['best']['step'],
                 extrapolation_accuracy=extra['accuracy'],phase_zero_accuracy=zero,
                 parameters=audit['total_parameters'],checkpoint_sha256=r['checkpoint_sha256'],**field_audit)
        rows.append(row)
    assert all(len(v)==1 for v in shared.values())
    assert all(len(v)==1 for v in dynamic.values())
    assert all(len(v)==1 for v in tensor_hashes.values())
    groups=[]
    for arm in dict.fromkeys(t['arm'] for t in m['trials']):
        rs=sorted((r for r in rows if r['arm']==arm),key=lambda r:r['seed'])
        expected=sorted(t['seed'] for t in m['trials'] if t['arm']==arm)
        assert [r['seed'] for r in rs]==expected
        vals=[100*r['accuracy'] for r in rs]
        groups.append(dict(arm=arm,seeds=expected,n=len(vals),mean=statistics.mean(vals),
                           sd=statistics.stdev(vals),seed_values=vals,
                           extrapolation_seed_values=[100*r['extrapolation_accuracy'] for r in rs],
                           phase_zero_seed_values=[None if r['phase_zero_accuracy'] is None else 100*r['phase_zero_accuracy'] for r in rs]))
    summary=dict(groups=groups,rows=rows,verified_checkpoints=len(rows),
                 source_verified=True,validation_selection_replayed=True,
                 shared_initialization_verified=True,full_dynamic_initialization_verified=True,
                 primary_and_extrapolation_predictions_replayed=True)
    (root/'verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'seed-results.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
    lines=['# Anonymous inductive graph classification','',
           '| Field | Test accuracy, mean ± sample SD (%) | Every fixed seed (%) |',
           '|---|---:|---|']
    for g in groups:
        lines.append(f"| {g['arm']} | {g['mean']:.3f} ± {g['sd']:.3f} | "+', '.join(f'{v:.3f}' for v in g['seed_values'])+' |')
    lines+=['',f'All {len(rows)} checkpoints, validation selections, initializations and predictions replayed.',
            '', 'Scope: one graph-aware K=2 Q/K transport block followed by a complete global attention network and invariant classifier. Deterministic topology-only equivariant fixed fields are covered by the obstruction; random symmetry-breaking marks and input-dependent nonlocal Flat fields are separate controls. No arbitrary-network or universal-circulation-necessity claim.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
