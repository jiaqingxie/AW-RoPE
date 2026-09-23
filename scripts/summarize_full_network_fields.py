"""Replay every frozen full-network checkpoint and expose stronger controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();manifest=json.loads((root/'manifest.json').read_text())
    source=Path(manifest['source']);sys.path.insert(0,str(source/'scripts'))
    import torch
    import numpy as np
    import full_network_field_prediction as f
    torch.set_num_threads(1)
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    rows=[];common={};field_initial={};datasets={}
    for t in manifest['trials']:
        rd=root/t['id'];result=json.loads((rd/'result.json').read_text())
        audit=json.loads((rd/'audit.json').read_text())
        assert f.r.sha(rd/'best.pt')==result['checkpoint_sha256']
        trajectory=[json.loads(s) for s in (rd/'trajectory.jsonl').read_text().splitlines()]
        selected=min(trajectory,key=lambda r:r['validation']['bce'])
        assert selected==result['best']
        d=f.generate(256,t['data_seed']+2,t['task'])
        assert f.r.state_hash(d)==result['test_tensor_sha256']
        torch.manual_seed(t['seed']);model=f.Model(t['arm'],task=t['task'],seed=t['seed'],layers=t['layers'],gain=t['gain'])
        assert model.audit['initial_sha256']==audit['initial_sha256']
        common.setdefault((t['task'],t['seed']),set()).add(audit['shared_initial_sha256'])
        if t['arm'] in ('unrestricted','gradient','projected-flat'):
            field_initial.setdefault((t['task'],t['seed']),set()).add(audit['initial_sha256'])
        datasets.setdefault(t['task'],set()).add(result['test_tensor_sha256'])
        model.load_state_dict(torch.load(rd/'best.pt',weights_only=True))
        metrics,logits=f.evaluate(model,d)
        assert metrics==result['test']
        saved=np.load(rd/'predictions.npz');np.testing.assert_array_equal(saved['logits'],logits.numpy());np.testing.assert_array_equal(saved['labels'],d['y'].numpy())
        cycle_sums=[]
        if t['arm'] in ('unrestricted','gradient','projected-flat'):
            import inspect
            original=model.pos[0]._transport_single;sig=inspect.signature(original)
            def capture(*a,**kw):
                displacement=sig.bind(*a,**kw).arguments['displacement']
                cycle_sums.append(displacement.reshape(-1,24)[:,::2].sum(-1).detach())
                return original(*a,**kw)
            model.pos[0]._transport_single=capture
            with torch.no_grad():model(f.batch(d,slice(0,128)))
            model.pos[0]._transport_single=original
        if t['arm'] in ('wire','wire-mixing'):
            # The training runner's generic intervention flag does not zero
            # WIRE. Perform its actual zero-angle intervention explicitly here.
            with torch.no_grad():
                for pos in model.pos:pos.weight.zero_()
            zero=f.evaluate(model,d)[0]
        else:
            zero=f.evaluate(model,d,intervention='zero-phase')[0]
            assert zero==result['test_zero_phase']
        row=dict(task=t['task'],arm=t['arm'],seed=t['seed'],test_accuracy=metrics['accuracy'],
                 validation_accuracy=result['validation']['accuracy'],best_step=selected['step'],
                 phase_zero_test_accuracy=zero['accuracy'],max_pair_logit_gap=metrics['max_pair_logit_gap'],
                 parameters=audit['total_parameters'],checkpoint_sha256=result['checkpoint_sha256'])
        if cycle_sums:
            c=torch.cat(cycle_sums);row['max_abs_cycle_sum']=float(c.abs().max())
            row['max_paired_cycle_sum_difference']=float((c[1::2]-c[::2]).abs().max())
            if t['arm'] in ('gradient','projected-flat'):assert row['max_abs_cycle_sum']<1e-9
        rows.append(row)
    assert all(len(v)==1 for v in common.values())
    assert all(len(v)==1 for v in field_initial.values())
    assert all(len(v)==1 for v in datasets.values())
    groups=[]
    for task,arm in dict.fromkeys((r['task'],r['arm']) for r in rows):
        rs=[r for r in rows if (r['task'],r['arm'])==(task,arm)]
        assert sorted(r['seed'] for r in rs)==list(range(5))
        values=[100*r['test_accuracy'] for r in rs]
        groups.append(dict(task=task,arm=arm,n=5,mean=statistics.mean(values),sd=statistics.stdev(values),
                           seed_values=values,phase_zero_seed_values=[100*r['phase_zero_test_accuracy'] for r in rs]))
    summary=dict(groups=groups,rows=rows,verified_checkpoints=len(rows),source_verified=True,
                 common_initialization_verified=True,full_dynamic_initialization_verified=True,
                 selection_replayed=True,all_logits_replayed=True,
                 scope='one global attention block with trainable encoder, residual/norm, FFN and mean-pool classifier; not arbitrary full networks',
                 wire_intervention_note='Use replayed explicit zero Omega values; generic flag in raw training result did not alter WIRE angles.')
    (root/'verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'seed-results.csv').open('w') as out:
        keys=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# Full-network field results','',f'All {len(rows)} checkpoints, validation selections, predictions and initialization identities verified.','',
           '| Task | Field | Test accuracy (%) | Seeds 0–4 (%) |','|---|---|---:|---|']
    for g in groups:lines.append(f"| {g['task']} | {g['arm']} | {g['mean']:.2f} ± {g['sd']:.2f} | "+', '.join(f'{v:.2f}' for v in g['seed_values'])+' |')
    lines += ['', 'C.6 Static and its fixed Flat projection are exactly Zero on this cycle and share that row; they are not independent fits.',
              '', 'The learned fixed template has stable canonical edge identities. The projected-flat field is input dependent and nonlocal, with cycle sums numerically zero.',
              '', 'These stronger controls rule out interpreting AW > Zero/local-gradient as proof that input dependence and nonzero circulation are both necessary.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2),flush=True)


if __name__=='__main__':main()
