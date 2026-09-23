"""Replay Exact route checkpoints and compare against the original fixed seeds."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import route_phase_exact_prediction as exact


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);run=args.run.resolve()
    protocol=json.loads((run/'protocol.json').read_text())
    reference=Path(protocol['reference_protocol']).parent
    for name,digest in protocol['source_sha256'].items():
        assert exact.base.sha(run/'source'/name)==digest
        assert exact.base.sha(Path(__file__).resolve().parents[1]/name)==digest
    datasets={split:exact.base.generate(protocol[key],protocol['data_seed']+offset,
              route_length=protocol['route_length'],motifs=protocol['motifs'])
              for split,key,offset in [('train','train_pairs',0),('validation','val_pairs',1),('test','test_pairs',2)]}
    fingerprints={split:{exact.base.state_hash({'x':x}) for x in data['x']} for split,data in datasets.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert fingerprints[a].isdisjoint(fingerprints[b])
    records=[];summary={}
    for readout in protocol['readouts']:
        rows=[]
        for seed in protocol['seeds']:
            cell=run/readout/'exact-aw'/f'seed-{seed}'
            result=json.loads((cell/'result.json').read_text())
            audit=json.loads((cell/'audit.json').read_text())
            prior=json.loads((reference/readout/'aw'/f'seed-{seed}'/'result.json').read_text())
            for key in ['data_seed','route_length','motifs','depth','z','steps','lr',
                        'train_pairs','val_pairs','test_pairs','batch_pairs','layers','width']:
                assert result['config'][key]==protocol[key]==prior['config'][key]
            assert result['parameter_audit']==prior['parameter_audit']
            assert result['checkpoint_sha256']==exact.base.sha(cell/'best.pt')
            assert result['trajectory_sha256']==exact.base.sha(cell/'trajectory.jsonl')
            assert result['source_hashes']==prior['source_hashes']
            assert result['exact_extension']['extension_sha256']==protocol['source_sha256']['scripts/route_phase_exact_prediction.py']
            assert result['exact_extension']['exact_operator_sha256']==protocol['source_sha256']['scripts/exact_aw_operator.py']
            for split in ['train','validation']:
                assert audit[split+'_tensor_sha256']==datasets[split]['tensor_sha256']
            assert result['test_tensor_sha256']==datasets['test']['tensor_sha256']==prior['test_tensor_sha256']
            trajectory=[json.loads(line) for line in (cell/'trajectory.jsonl').read_text().splitlines()]
            assert [t['step'] for t in trajectory]==list(range(0,protocol['steps']+1,protocol['eval_every']))
            assert min(trajectory,key=lambda t:t['validation']['loss'])==result['best']
            cp=torch.load(cell/'best.pt',map_location='cpu',weights_only=False)
            assert cp['step']==result['best']['step']
            torch.manual_seed(seed)
            model=exact.ExactRoutePredictor(readout=readout,d=protocol['width'],L=protocol['depth'],
                                           z=protocol['z'],layers=protocol['layers'],seed=seed)
            model.load_state_dict(cp['model'])
            metrics,logits=exact.base.evaluate(model,datasets['test'],return_logits=True)
            saved=np.load(cell/'test-predictions.npz')
            torch.testing.assert_close(logits,torch.from_numpy(saved['logits']),atol=0,rtol=0)
            assert np.array_equal(saved['labels'],datasets['test']['y'].numpy())
            assert metrics==result['test']
            zero=exact.base.evaluate(model,datasets['test'],intervention='zero-phase')
            assert zero==result['test_zero_phase']
            row=dict(readout=readout,seed=seed,test=metrics,zero_phase=zero,
                     sparse_test=prior['test'],best_step=result['best']['step'],
                     parameter_audit=result['parameter_audit'],checkpoint_sha256=result['checkpoint_sha256'],
                     trajectory_sha256=result['trajectory_sha256'])
            if readout=='endpoint':
                dd={k:v.double() if v.is_floating_point() else v for k,v in exact.base.batch(datasets['test'],slice(None)).items()}
                _,dl=exact.base.evaluate(model.double(),dd,return_logits=True,intervention='zero-phase')
                gap=float((dl[::2]-dl[1::2]).abs().max())
                assert gap<1e-9
                row['zero_phase_float64_max_pair_logit_gap']=gap
            rows.append(row);records.append(row)
        summary[readout]={}
        for key in ['test','zero_phase','sparse_test']:
            values=[100*row[key]['accuracy'] for row in rows]
            summary[readout][key]=dict(mean_percent=statistics.mean(values),sample_sd_percent=statistics.stdev(values),seeds_percent=values)
    result=dict(protocol=protocol,summary=summary,records=records,audit=dict(completed_runs=10,
                test_predictions_reproduced=True,parameter_counts_and_initializations_match_sparse=True,
                train_validation_test_fingerprints_match_sparse=True,raw_split_overlap=False,
                source_and_checkpoint_hashes_verified=True,validation_selection_verified=True,
                zero_phase_uses_exact_operator=True))
    args.out.mkdir(parents=True,exist_ok=True)
    exact.base.atomic_json(args.out/'summary.json',result)
    lines=['# Exact AW: constructed route-interference task','','Mean ± sample SD (%), seeds 0–4; checkpoint selected by minimum validation BCE.','',
           '| Method | Endpoint | Attention |','|---|---:|---:|']
    for label,key in [('Sparse AW (existing)','sparse_test'),('Exact AW','test'),('Exact AW, phases zeroed','zero_phase')]:
        cells=[summary[x][key] for x in ['endpoint','attention']]
        lines.append('| '+label+' | '+' | '.join(f"{c['mean_percent']:.2f} ± {c['sample_sd_percent']:.2f}" for c in cells)+' |')
    lines+=['','Exact uses the complete production resolvent. The two variants are independently trained from the same initialization and data, with equal budgets; no fitted-parameter equivalence is claimed.','',
            'Zeroing phases is an intervention on each selected Exact checkpoint, retaining the complete resolvent. It is not an independently trained Exact phase-free baseline.']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,audit=result['audit']),indent=2))


if __name__=='__main__':main()
