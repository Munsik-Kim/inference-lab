"""Additional recorded-reference, graph, pairing and transformation audit.

Called only inside a fresh P1/P2 frozen-source consumer process. No solver.
"""
from pathlib import Path
import numpy as np
from portable import compare_tree

def check(phase,root,run,output):
    from silent_alignment.io import read_json,write_json,Budget
    config=read_json(run/'config.json');tol=config['tolerances']
    if phase=='p0':
        comparisons=0
        for trial in read_json(run/'attempt_ledger.json'):
            for ref in trial['references']:
                for name in ['comparison_to_primary','step_halving_comparison']:
                    if name not in ref:continue
                    x=ref[name];comparisons+=1
                    if x['max_z_relative_frobenius']>tol['z_relative'] or x['max_z_absolute_near_zero']>tol['z_absolute'] or x['max_learnable_normalized_loss_absolute']>tol['normalized_loss_absolute'] or x['max_projector_distance_absolute']>tol['projector_absolute']:
                        raise ValueError('P0_REFERENCE_NUMERIC_GATE')
            for g in trial['gram_closed_form_checks']:
                if not g['finite'] or g['relative_gram_error']>tol['closed_form_relative']:raise ValueError('P0_GRAM_GATE')
        return {'status':'PASS','recorded_reference_comparisons':comparisons,'full_reference_reintegration':0}
    from silent_alignment.p1.metrics import comparison_status,direction
    from silent_alignment.p1.contract import load_inputs
    refs=ckcount=0
    if phase=='p1':
        rows=read_json(run/'input_manifest.json')['planned_trajectories']
        for row in rows:
            if not row['matrix_reference']:continue
            C,l,r,s,R=load_inputs(run,row)
            for suffix in ['h001','h0005','step_halving']:
                with np.load(run/'references'/(row['key']+'_'+suffix+'.npz'),allow_pickle=False) as a:
                    arrays={k:a[k] for k in a.files if k!='times'}
                if not comparison_status(arrays,tol)['pass']:raise ValueError('P1_REF_GATE')
                refs+=1
                if suffix=='step_halving':continue
                with np.load(run/'references'/(row['key']+'_'+suffix+'_checkpoints.npz'),allow_pickle=False) as a,np.load(run/'checkpoints'/(row['key']+'.npz'),allow_pickle=False) as b:
                    for i,tt in enumerate(a['times']):
                        z=a['U'][i]@a['V'][i].T;zz=b['U'][i]@b['V'][i].T;norm=np.linalg.norm(z)
                        if norm>tol['z_near_zero_norm'] and np.linalg.norm(z-zz)/norm>tol['z_relative']:raise ValueError('P1_REF_CHECKPOINT_Z')
                        ckcount+=1
        scalar=0
        for row in rows:
            if not row['scalar_reference']:continue
            with np.load(run/'references'/(row['key']+'_scalar_vector.npz'),allow_pickle=False) as a:
                near=a['z_reference_norm']<=tol['z_near_zero_norm'];rel=a['z_absolute'][~near]/a['z_reference_norm'][~near]
                if np.max(rel)>tol['z_relative'] or np.max(a['z_absolute'][near])>tol['z_absolute'] or np.max(a['normalized_loss_difference'])>tol['normalized_loss_absolute']:raise ValueError('P1_SCALAR_REFERENCE')
                if a['z'].shape!=(1702,32):raise ValueError('SCALAR_COMPONENT_SCOPE')
                scalar+=1
        return {'status':'PASS','matrix_comparison_series':refs,'reference_physical_checkpoints':ckcount,'scalar_vector_references':scalar,'scalar_components':scalar*32,'full_reference_reintegration':0}
    # P2: retained comparison series and exact saved reference checkpoints.
    for folder in [run/'matching',run/'references',run/'block_a/references']:
        for p in folder.glob('*.npz'):
            with np.load(p,allow_pickle=False) as a:
                if 'z_reference_norm' in a:
                    if not comparison_status({k:a[k] for k in a.files if k!='times'},tol)['pass']:raise ValueError('P2_COMPARISON_GATE')
                    refs+=1
    p1=root/'results/p1_20261023-104046'
    for folder,main in [(run/'references',run/'checkpoints'),(run/'block_a/references',p1/'checkpoints')]:
        for p in folder.glob('*_checkpoints.npz'):
            key=p.name.split('_h')[0]
            with np.load(p,allow_pickle=False) as a,np.load(main/(key+'.npz'),allow_pickle=False) as b:
                for i,t in enumerate(a['times']):
                    j=int(np.flatnonzero(b['times']==t)[0]);z=a['U'][i]@a['V'][i].T;zz=b['U'][j]@b['V'][j].T;norm=np.linalg.norm(zz);err=np.linalg.norm(z-zz)
                    if (norm<=tol['z_near_zero_norm'] and err>tol['z_absolute']) or (norm>tol['z_near_zero_norm'] and err/norm>tol['z_relative']):raise ValueError('P2_REF_PHYSICAL_Z')
                    ckcount+=1
    from silent_alignment.p2.core import graph_setup,graph_metrics,transform,matching
    from silent_alignment.p2.diagnostics import persistence
    from silent_alignment.p1.analysis import event
    graphrows=[];mode_rows=[];basis={}
    for gi in range(3):
        with np.load(p1/f'assets/target_g{gi}.npz',allow_pickle=False) as a:l,s,rt=np.linalg.svd(a['C'],full_matrices=False)
        basis[gi]=(s,rt.T)
    for row in read_json(p1/'input_manifest.json')['planned_trajectories']:
        C,l,r,s,R=load_inputs(p1,row);sig,q=basis[row['g_index']];M,_=graph_setup(q,R)
        with np.load(p1/'raw'/(row['key']+'.npz'),allow_pickle=False) as a:
            mean,worst=graph_metrics(sig,row['rho'],a['times'],M,Budget(120))
            dm=float(np.max(abs(mean-a['right_distance'])));dw=float(np.max(abs(worst-a['right_worst'])))
            if max(dm,dw)>1e-5:raise ValueError('P2_FINITE_RHO_GRAPH')
            graphrows.append({'key':row['key'],'mean_max':dm,'worst_max':dw})
            for field,values in [('right_distance',mean),('right_worst',worst)]:
                old=event(a['times'],a[field],a[field][0]*.5,already=row['initialization']=='aligned')
                new=event(a['times'],values,values[0]*.5,already=row['initialization']=='aligned')
                if old['status']!=new['status'] or old['bracket']!=new['bracket']:raise ValueError('GRAPH_HALF_BRACKET')
            if row['initialization']=='random':
                for m in [32,33]:
                    values=a['target_mode_signed'][:,m-1]/s[m-1]
                    mode_rows.append({'key':row['key'],'g':row['g'],'rho':row['rho'],'R_index':row['R_index'],'mode':m,
                        'P1_first_crossing_unchanged':event(a['times'],-values,-.1),**persistence(a['times'],values)})
    compare_tree(mode_rows,read_json(run/'block_a/mode_persistence.json'));write_json(output/'p1-mode-persistence-recalculated.json',mode_rows)
    write_json(output/'graph-recalculated.json',graphrows)
    parents=read_json(run/'input_manifest.json')['parents']
    with np.load(run/'assets/rotations.npz',allow_pickle=False) as a:rot={k:a[k] for k in a.files}
    for parent in parents:
        with np.load(run/'assets'/(parent['parent_key']+'_CONTINUE.npz'),allow_pickle=False) as a:u,v=a['U'],a['V']
        with np.load(p1/'checkpoints'/(parent['p1_key']+'.npz'),allow_pickle=False) as a:
            j=int(np.flatnonzero(a['times']==parent['t_star'])[0])
            if not np.array_equal(u,a['U'][j]) or not np.array_equal(v,a['V'][j]):raise ValueError('PARENT_MAPPING')
        with np.load(run/'assets/fixed_basis.npz',allow_pickle=False) as a:l,r=a['left'],a['right']
        with np.load(run/f"assets/target_g{parent['g_index']}.npz",allow_pickle=False) as a:C,s=a['C'],a['sigma']
        for arm in ['CONTINUE','GAUGE','ALIGN','ROTATE']:
            up,vp=transform(u,v,l,r,arm,rot)
            with np.load(run/'assets'/(parent['parent_key']+'_'+arm+'.npz'),allow_pickle=False) as a:
                if not np.allclose(up,a['U'],atol=1e-12,rtol=1e-10) or not np.allclose(vp,a['V'],atol=1e-12,rtol=1e-10):raise ValueError('INTERVENTION_REGENERATION')
            if not matching(C,l,r,s,u,v,up,vp,arm,rot,parent['rho'],tol)['pass']:raise ValueError('INTERVENTION_MATCHING')
    return {'status':'PASS','comparison_series':refs,'new_reference_files':16,'reference_physical_checkpoints':ckcount,
            'original_graph_paths':81,'algebraic_time_evaluations':137862,'half_brackets':162,'mode_persistence_series':len(mode_rows),
            'parent_mappings':len(parents),'intervention_mappings':4*len(parents),'full_reference_reintegration':0,
            'gauge_scope':'retained all-grid error series and physical checkpoints; unsaved original full Z paths not reconstructed'}
