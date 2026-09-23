"""Verify every frozen run and create portable route-parity results and figures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import route_phase_prediction as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads((args.run/'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        assert r.sha(args.run/'source'/name) == expected, name
    assert r.sha(r.LAYER) == protocol['source_sha256']['external/Graph-RoPE/graphgps/layer/aw_rope.py']
    assert r.sha(r.__file__) == protocol['source_sha256']['scripts/route_phase_prediction.py']
    datasets = {split: r.generate(protocol[key], protocol['data_seed']+offset,
               route_length=protocol['route_length'], motifs=protocol['motifs'])
               for split, key, offset in [('train','train_pairs',0),('validation','val_pairs',1),('test','test_pairs',2)]}
    sample_hashes = {split: {r.state_hash({'x':x}) for x in d['x']} for split,d in datasets.items()}
    assert all(len(sample_hashes[s]) == len(datasets[s]['y']) for s in datasets)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert sample_hashes[a].isdisjoint(sample_hashes[b]), (a,b)
    records, table = [], {}
    for readout in protocol['readouts']:
        table[readout] = {}
        for arm in protocol['arms']:
            rows = []
            for seed in protocol['seeds']:
                cell = args.run/readout/arm/f'seed-{seed}'
                result = json.loads((cell/'result.json').read_text())
                audit = json.loads((cell/'audit.json').read_text())
                config = result['config']
                for key in ['data_seed','route_length','motifs','depth','z','steps','lr',
                            'train_pairs','val_pairs','test_pairs','batch_pairs','layers','width']:
                    assert config[key] == protocol[key], (cell,key)
                assert (config['arm'],config['readout'],config['seed']) == (arm,readout,seed)
                assert r.sha(cell/'best.pt') == result['checkpoint_sha256']
                assert r.sha(cell/'trajectory.jsonl') == result['trajectory_sha256']
                trajectory = [json.loads(line) for line in (cell/'trajectory.jsonl').read_text().splitlines()]
                assert [t['step'] for t in trajectory] == list(range(0,protocol['steps']+1,protocol['eval_every']))
                best = min(trajectory, key=lambda t:t['validation']['loss'])
                assert best == result['best']
                state = torch.load(cell/'best.pt',map_location='cpu',weights_only=False)
                assert state['step'] == best['step']
                for source in ('experiment','production_aw'):
                    assert state['source_hashes'][source] == result['source_hashes'][source]
                assert result['source_hashes']['experiment'] == r.sha(r.__file__)
                for split in ('train','validation'):
                    assert audit[split+'_tensor_sha256'] == datasets[split]['tensor_sha256']
                assert result['test_tensor_sha256'] == datasets['test']['tensor_sha256']
                predictions = np.load(cell/'test-predictions.npz')
                assert np.array_equal(predictions['labels'],datasets['test']['y'].numpy())
                acc = float(((predictions['logits']>=0)==predictions['labels'].astype(bool)).mean())
                assert acc == result['test']['accuracy']
                torch.manual_seed(seed)
                model = r.RoutePredictor(arm,readout=readout,d=protocol['width'],L=protocol['depth'],
                                         z=protocol['z'],layers=protocol['layers'],seed=seed)
                assert model.parameter_audit == result['parameter_audit']
                model.load_state_dict(state['model'])
                reproduced, logits = r.evaluate(model,datasets['test'],return_logits=True)
                torch.testing.assert_close(logits,torch.from_numpy(predictions['logits']),atol=0,rtol=0)
                assert reproduced == result['test']
                row = {'readout':readout,'arm':arm,'seed':seed,'test':result['test'],
                       'best_step':best['step'],'parameters':result['parameter_audit'],
                       'checkpoint_sha256':result['checkpoint_sha256'],
                       'trajectory_sha256':result['trajectory_sha256']}
                if arm in ('aw','gradient'):
                    assert r.evaluate(model,datasets['test'],intervention='zero-phase') == result['test_zero_phase']
                    row['test_zero_phase'] = result['test_zero_phase']
                    sample = r.batch(datasets['test'],slice(0,128))
                    with torch.no_grad():
                        h = model.adapters[0](model.encoder(sample['x']).flatten(0,1))
                        phase = model.pos[0].edge_field(h,r.flatten_edges(sample['edges'],sample['x'].shape[1]))
                        circulation = phase.reshape(128,-1)[:,::2].sum(-1)
                    row['mean_abs_learned_cycle_circulation'] = circulation.abs().mean().item()
                    row['max_abs_learned_cycle_circulation'] = circulation.abs().max().item()
                if readout == 'endpoint':
                    double_data = {k:v.double() if v.is_floating_point() else v for k,v in r.batch(datasets['test'],slice(None)).items()}
                    _, check_logits = r.evaluate(model.double(), double_data, return_logits=True,
                                                intervention='zero-phase' if arm=='aw' else None)
                    row['float64_collision_max_gap'] = (check_logits[::2]-check_logits[1::2]).abs().max().item()
                    row['float64_collision_accuracy'] = ((check_logits>=0)==double_data['y'].bool()).double().mean().item()
                    assert row['float64_collision_max_gap'] < 1e-9
                rows.append(row); records.append(row)
            vals = [100*row['test']['accuracy'] for row in rows]
            table[readout][arm] = {'mean_percent':statistics.mean(vals),
                                  'sample_sd_percent':statistics.stdev(vals),'seeds_percent':vals}
        for seed in protocol['seeds']:
            selected = [x for x in records if x['seed']==seed and x['readout']==readout]
            assert len({x['parameters']['total'] for x in selected}) == 1
            assert len({x['parameters']['shared_initial_hash'] for x in selected}) == 1
    result = {'protocol':protocol, 'summary':table, 'records':records,
              'audit':{'completed_runs':len(records),'test_predictions_reproduced':True,
                       'shared_initializations_verified':True,'trainable_counts_matched':True,
                       'raw_split_overlap':False,'source_and_checkpoint_hashes_verified':True,
                       'selection_reproduced_from_validation_trajectories':True}}
    args.out.mkdir(parents=True,exist_ok=True)
    r.atomic_json(args.out/'summary.json',result)
    lines = ['# Route-parity prediction: completed matched study', '',
             'Test accuracy in percent, mean ± sample SD over all five fixed seeds.', '',
             '| Readout | WIRE | Mixing | WIRE + mixing | AW | Gradient field |',
             '|---|---:|---:|---:|---:|---:|']
    for readout, cells in table.items():
        lines.append('| '+readout+' | '+' | '.join(f"{cells[a]['mean_percent']:.2f} ± {cells[a]['sample_sd_percent']:.2f}" for a in protocol['arms'])+' |')
    lines += ['', 'All four primary arms see the same raw node features and graph-only spectral coordinates. AW learns its edge field from encoded node features. It receives no target phase.',
              '', 'The endpoint result proves a separation for pointwise encoding followed by one linear transport and endpoint readout. It does not establish impossibility for multi-layer message passing or global attention. The attention panel tests the latter empirically.',
              '', 'The full attention block matches the production GraphRoPE forward and VJP in float64 tests. This is a small single-head, single-block graph Transformer, not a rerun of the larger benchmark GraphGPS configuration.',
              '', 'A zero-phase intervention is applied to each frozen AW checkpoint without retraining. This measures reliance on its learned phases; it is not an independent phase-free model.',
              '', 'The old 95%/50% two-example toy used unequal information and no held-out split. It is excluded. Development pilots and every formal seed remain on disk.']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    tex = []
    for readout,cells in table.items():
        tex.append(('Endpoint' if readout=='endpoint' else 'Attention')+' & '+' & '.join(
            f"{cells[a]['mean_percent']:.2f}\\,$\\pm$\\,{cells[a]['sample_sd_percent']:.2f}" for a in protocol['arms'])+r' \\')
    (args.out/'rows.tex').write_text('\n'.join(tex)+'\n'+r'\bottomrule'+'\n')
    draw(table,args.out,datasets['test'])
    print(json.dumps({'summary':table,'audit':result['audit']},indent=2))


def draw(table,out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(9.0,5.6),gridspec_kw={'height_ratios':[1,1.3]},layout='constrained')
    n = data['x'].shape[1]; m=n//2
    coordinates = np.zeros((n,2))
    for p in range(m+1):
        coordinates[p] = [p/m, np.sin(np.pi*p/m)]
        if 0<p<m: coordinates[n-p] = [p/m,-np.sin(np.pi*p/m)]
    inverse = torch.argsort(torch.tensor(data['metadata'][0]['permutation']))
    for label,ax in enumerate(axes[0]):
        x = data['x'][label,inverse].numpy()
        for i in range(n):
            v = (i+1)%n
            ax.plot(coordinates[[i,v],0],coordinates[[i,v],1],color='#707070',lw=1.8,zorder=1)
        colors = ['#e8e8e8']*n
        for i in range(n):
            if x[i,2]>0: colors[i]='#E89B35'
            if x[i,3]>0: colors[i]='#427FB5'
            if x[i,0]>0 or x[i,1]>0: colors[i]='#333333'
        ax.scatter(coordinates[:,0],coordinates[:,1],c=colors,s=190,edgecolors='white',linewidths=1,zorder=2)
        for i in range(n):
            if x[i,2]>0 or x[i,3]>0:
                ax.text(coordinates[i,0],coordinates[i,1], 'A' if x[i,2]>0 else 'B',
                        color='white',ha='center',va='center',fontsize=9.5,weight='bold')
        ax.text(-.06,0,'u',ha='center',va='center'); ax.text(1.06,0,'v',ha='center',va='center')
        ax.set_title(['Class 0: both motifs on one route','Class 1: motifs split across routes'][label],fontsize=11)
        ax.set_xlim(-.13,1.13); ax.set_ylim(-1.3,1.3); ax.axis('off')
    names=['WIRE','Mixing','WIRE +\nmixing','AW','Gradient']
    palette=['#92999F','#92999F','#92999F','#347F77','#B5BBC0']
    for ax,readout in zip(axes[1],('endpoint','attention')):
        for i,arm in enumerate(('wire','mixing','wire-mixing','aw','gradient')):
            row=table[readout][arm]
            ax.bar(i,row['mean_percent'],color=palette[i],width=.67)
            ax.errorbar(i,row['mean_percent'],yerr=row['sample_sd_percent'],color='#202020',capsize=3,lw=1.4)
            ax.text(i,row['mean_percent']+row['sample_sd_percent']+2,f"{row['mean_percent']:.1f}",ha='center',fontsize=9)
        ax.axhline(50,color='#595959',ls='--',lw=1)
        ax.set_xticks(range(5),names,fontsize=9); ax.set_ylim(0,114)
        ax.set_yticks([0,25,50,75,100]); ax.set_ylabel('Test accuracy (%)')
        ax.set_title('Endpoint readout' if readout=='endpoint' else 'Global attention readout',fontsize=11)
    fig.savefig(out/'route_parity.pdf',bbox_inches='tight')
    fig.savefig(out/'route_parity.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
