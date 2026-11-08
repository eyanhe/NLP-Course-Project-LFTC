"""Pure resolved-run and predeclared statistical registries (design §§5–9).

These builders read no files, train no models and create no locks. Callers supply
actual anchor choices, actual completed D-pilot metadata and prepared identities.
The freeze layer separately verifies their saved evidence and human reviews.

``asset_hash`` is the prepared asset identity; ``selection_ids_hashes`` contains
SHA256('\\n'.join(sorted(select_ids))) for each dataset. ``diagnostics_code_hash``
is the caller's combined mechanism implementation identity, separate from the
pinned core learner hash. Neither learned features nor runtime-resource request
fingerprints are predicted here.

Contrast output follows research_v3.statistics's schema. F6 evaluation view IDs
identify future paired predictions from one checkpoint, not additional fits.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from typing import Any

from .config import canonical_hash
from .types import RunSpec

EMPTY_IDS_HASH = hashlib.sha256(b'').hexdigest()
NEURAL = {'bilstm','distilbert','modernbert','m2_distilbert','frozen_generic_distilbert_head'}
MECHANISM_MODELS = {'m2_distilbert','fixed_tfidf_svm','frozen_generic_distilbert_head'}


def _metadata(prepared):
    return prepared.get('datasets', prepared)


def _choices(design, selections, *, require_pilots=True):
    choices = selections.get('choices', selections)
    for node in design['selection_nodes']:
        choice = choices.get(node['node_id'], {})
        if choice.get('status') != 'complete':
            raise PermissionError(f"Unresolved anchor selection: {node['node_id']}")
        if choice.get('hparams') not in node['candidates']:
            raise ValueError(f"Selection is not a declared candidate: {node['node_id']}")
        if not choice.get('evidence_sha256'):
            raise ValueError(f"Selection evidence is missing: {node['node_id']}")
    for node in design['discovery_pilot_nodes'] if require_pilots else []:
        if selections.get('discovery_pilots', {}).get(node['pilot_id'], {}).get('status') != 'complete':
            raise PermissionError(f"Unresolved real discovery pilot: {node['pilot_id']}")
    return choices


def _protocol(design, protocol):
    identity = canonical_hash(protocol)
    if identity != design['protocol_hash']:
        raise ValueError('Protocol changed after the declared design')
    return identity


def build_run_manifest(design: dict[str, Any], selections: dict[str, Any],
                       prepared_metadata: dict[str, Any], protocol: dict[str, Any],
                       core_code_hash: str, diagnostics_code_hash: str,
                       selection_ids_hashes: dict[str, str], *, asset_hash: str,
                       training_only: bool=False) -> dict[str, Any]:
    """Resolve every planned physical ID once without resampling its condition.

    Supports the actual selections.json envelope (``choices`` and
    ``discovery_pilots``), or a direct choice mapping with discovery_pilots.
    status='resolved' means metadata dependencies are supplied; checkpoint and
    human evidence verification remains mandatory in the global freeze.
    """
    if training_only and selections.get('selection_policy')!='fixed_defaults':
        raise PermissionError('Training before pilots requires the user-fixed parameter policy')
    choices = _choices(design,selections,require_pilots=not training_only)
    protocol_hash = _protocol(design,protocol)
    if not asset_hash or not core_code_hash or not diagnostics_code_hash:
        raise ValueError('Asset, pinned core and diagnostics code identities are required')
    metadata = _metadata(prepared_metadata)
    conditions = {c['condition_id']:c for c in design['conditions']}
    selection_nodes = {n['node_id']:n for n in design['selection_nodes']}
    aliases = defaultdict(list)
    for row in design['logical_runs']:
        aliases[row['run_id']].append(deepcopy(row))
    planned_ids = [r['run_id'] for r in design['physical_runs']]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError('Duplicate planned physical run ID')
    if set(aliases)-set(planned_ids):
        raise ValueError('Logical alias references an undeclared physical run')
    runs = []
    for physical in design['physical_runs']:
        condition = conditions[physical['condition_id']]
        ds = physical['dataset_id']; meta = metadata[ds]
        strict = physical['strict_no_val']; model = physical['model_id']
        binding = physical['hparam_binding']
        if strict and 'fixed' not in binding:
            raise ValueError('Strict E2 cannot bind anchor selection hyperparameters')
        selected = binding.get('fixed')
        if selected is None:
            selected = choices[binding['selection_node']]['hparams']
        select_hash = EMPTY_IDS_HASH if strict else selection_ids_hashes.get(ds)
        if not select_hash:
            raise ValueError(f'Missing selection IDs hash for {ds}')
        hp = {**deepcopy(physical['training_policy']),**deepcopy(selected),
              **deepcopy(physical['variant']),
              'role_ids_hash':{'train':condition['training_set_hash'],
                               'select':select_hash,'discovery':EMPTY_IDS_HASH},
              'target_task':meta['task'],
              **{k:protocol['runtime'][k] for k in ('device','micro_batch','cpu_threads')},
              'protocol_hash':protocol_hash,'label_count':len(meta['label_names'])}
        if not strict and meta['task'] in {'multilabel','multi_label'}:
            hp['select_ap_mask'] = deepcopy(meta['partitions']['select']['ap_mask'])
            if not hp['select_ap_mask']:
                raise ValueError('Multilabel selection AP mask is empty')
        if strict:
            hp.pop('select_ap_mask',None)
            if model in {'bilstm','distilbert','modernbert'}:
                expected = 10 if model=='bilstm' else 5
                if hp.get('epochs') != expected:
                    raise ValueError('Strict E2 requires its fixed epoch budget')
        diagnostics = model in MECHANISM_MODELS or (model=='bilstm' and bool(physical['variant']))
        engine = 'diagnostics' if diagnostics else 'core'
        code_hash = diagnostics_code_hash if diagnostics else core_code_hash
        seed = physical['training_seed'] if model in NEURAL else None
        spec = RunSpec('',condition['condition_id'],model,seed,hp,meta['inputs']['input_hash'],
                       asset_hash,code_hash,strict)
        spec = replace(spec,run_id=canonical_hash(asdict(spec)))
        if 'selection_node' in binding:
            node = selection_nodes[binding['selection_node']]
            selected_id = choices[node['node_id']].get('selected_run_id')
            if (selected_id and condition['condition_id']==node['anchor_condition_id']
                    and seed==node['training_seed'] and not physical['variant']
                    and selected_id!=spec.run_id):
                raise ValueError(f"Selected anchor checkpoint identity changed: {node['node_id']}")
        run = asdict(spec)
        runs.append({'planned_run_id':physical['run_id'],'run_id':spec.run_id,'artifact_id':spec.run_id,
                     'run':run,'engine':engine,'expected_model_request_hash':canonical_hash(run),
                     'condition_id':condition['condition_id'],'dataset_id':ds,'model_id':model,
                     'data_seed':condition['data_seed'],'training_seed':seed,
                     'semantic_aliases':aliases[physical['run_id']], 'status':'pending_checkpoint'})
    result = {'schema_version':1,'status':'training_resolved' if training_only else 'resolved','design_hash':design['design_hash'],
              'protocol_hash':protocol_hash,'runs':runs,'unresolved_nodes':[],
              'unique_fit_count':len({r['artifact_id'] for r in runs}),
              'planned_physical_run_count':len(runs),'logical_run_count':len(design['logical_runs']),
              'infeasible_conditions':deepcopy(design['infeasible_conditions']),
              'core_code_hash':core_code_hash,'diagnostics_code_hash':diagnostics_code_hash,
              'asset_hash':asset_hash,'actual_checkpoints_verified':False,
              'actual_human_pilots_verified':False,'official_test_predictions_generated':False,
              'training_only':training_only,'official_test_predictions_allowed':False,
              'pending_discovery_pilots':[n['pilot_id'] for n in design['discovery_pilot_nodes']
                  if selections.get('discovery_pilots',{}).get(n['pilot_id'],{}).get('status')!='complete']}
    result['manifest_hash'] = canonical_hash(result)
    return result


class _Contrasts:
    def __init__(self, design, manifest, prepared, protocol):
        self.design = design
        self.metadata = _metadata(prepared)
        self.protocol = protocol
        self.conditions = {c['condition_id']:c for c in design['conditions']}
        self.by_planned = {r['planned_run_id']:r for r in manifest['runs']}
        self.runs = {}
        self.estimates = []
        self.contrasts = []
        for r in manifest['runs']:
            self.runs[r['artifact_id']] = {k:r[k] for k in ('dataset_id','model_id','data_seed','training_seed')}

    def rows(self, experiment):
        # Preserve contexts even where one physical fit has several semantic uses.
        return [dict(logical,entry=self.by_planned[logical['run_id']],
                     condition=self.conditions[logical['condition_id']])
                for logical in self.design['logical_runs'] if logical['experiment_id']==experiment]

    def estimate(self, estimate_id, left, right, **metadata):
        if not left or not right:
            raise ValueError(f'Incomplete predeclared contrast arms: {estimate_id}')
        arms = []
        datasets = set()
        for records in (left,right):
            draws = defaultdict(set)
            for record in records:
                entry = record['entry']
                datasets.add(entry['dataset_id'])
                draws[entry['data_seed']].add(entry['artifact_id'])
            arms.append(draws)
        if len(datasets)!=1 or set(arms[0])!=set(arms[1]):
            raise ValueError('Contrast arms must share one dataset and actual data draws')
        ds = next(iter(datasets))
        metric = 'macro_ap' if self.metadata[ds]['task'] in {'multi_label','multilabel'} else 'macro_f1'
        estimate = {'estimate_id':estimate_id,'metric':metric,'dataset_id':ds,
                    'draws':[{'data_seed':seed,'left_runs':sorted(arms[0][seed]),
                              'right_runs':sorted(arms[1][seed])} for seed in sorted(arms[0])],**metadata}
        self.estimates.append(estimate)
        return estimate_id

    def contrast(self, contrast_id, family, components, kind='difference', **metadata):
        self.contrasts.append({'contrast_id':contrast_id,'family':family,'kind':kind,
                               'components':[{'estimate_id':e,'coefficient':c} for e,c in components],
                               **metadata})

    def pair(self, estimate_id, records, left_model='distilbert',right_model='tfidf_svm',**metadata):
        return self.estimate(estimate_id,[r for r in records if r['model_id']==left_model],
                             [r for r in records if r['model_id']==right_model],**metadata)

    def curves(self):
        for experiment,family in [('E1','F1'),('E2','F2'),('E4','F4'),('E5','F5')]:
            groups = defaultdict(list)
            for row in self.rows(experiment):
                meta = row['condition']['metadata']
                level = (meta['shots_per_class'] if experiment=='E2' else
                         row['context']['alpha'] if experiment=='E4' else meta['size'])
                groups[(row['dataset_id'],level)].append(row)
            by_dataset = defaultdict(list)
            for (ds,level),rows in groups.items():
                eid = f'{experiment}:{ds}:{level}:delta'
                self.pair(eid,rows,'modernbert' if experiment=='E5' else 'distilbert',
                          'distilbert' if experiment=='E5' else 'tfidf_svm',
                          experiment_id=experiment,level=level)
                self.contrast(eid,family,[(eid,1)],dataset_id=ds,level=level)
                by_dataset[ds].append((level,eid))
            for ds,levels in by_dataset.items():
                levels.sort(key=lambda x:float('inf') if x[0]=='full' else x[0])
                for (a,ea),(b,eb) in zip(levels,levels[1:]):
                    self.contrast(f'{experiment}:{ds}:{a}->{b}',family,[(eb,1),(ea,-1)],
                                  'interaction',dataset_id=ds,levels=[a,b])

    def long_tail(self):
        groups = defaultdict(list); heads = defaultdict(set)
        for row in self.rows('E3'):
            ds = row['dataset_id']; context = row['context']; ratio = context['ratio']
            order = tuple(context['head_order']) if ratio!=1 else ()
            groups[(ds,order,ratio)].append(row)
            if ratio!=1:
                heads[(ds,order)].add(context['head_seed'])
        estimates = {}
        for (ds,order,ratio),rows in groups.items():
            tag = '-'.join(map(str,order)) if order else 'shared'
            eid = f'E3:{ds}:head-{tag}:rho-{ratio}:delta'
            head_aliases = sorted(heads[(ds,order)]) if order else []
            self.pair(eid,rows,experiment_id='E3',level=ratio,head_order=list(order),
                      head_seed_aliases=head_aliases,
                      tail_definition='current_head_permutation' if order else 'NA_balanced',
                      tail_label_ids=list(order[-max(1,len(order)//3):]) if order else None)
            self.contrast(eid,'F3',[(eid,1)],dataset_id=ds,level=ratio,head_seed_aliases=head_aliases)
            estimates[(ds,order,ratio)] = eid
        for ds,order in heads:
            ratios = sorted(r for d,o,r in groups if d==ds and (o==order or not o))
            for a,b in zip(ratios,ratios[1:]):
                ea = estimates[(ds,() if a==1 else order,a)]
                eb = estimates[(ds,order,b)]
                tag = '-'.join(map(str,order))
                self.contrast(f'E3:{ds}:head-{tag}:{a}->{b}','F3',[(eb,1),(ea,-1)],
                              'interaction',dataset_id=ds,levels=[a,b],head_order=list(order),
                              head_seed_aliases=sorted(heads[(ds,order)]))

    def m2(self):
        groups = defaultdict(list)
        for row in self.rows('M2'):
            groups[(row['dataset_id'],row['condition']['metadata']['size'])].append(row)
        main_lr = self.protocol['experiments']['E6']['mechanism_rules']['M2']['encoder_lr']
        for (ds,size),rows in groups.items():
            prefix = f'M2:{ds}:{size}'
            arms = defaultdict(list)
            for row in rows:
                hp = row['entry']['run']['hparams']
                arms[(hp['arm'],hp['encoder_lr'])].append(row)
            def pair(name,left,right):
                eid = self.estimate(f'{prefix}:{name}',arms[(left,main_lr)],arms[(right,main_lr)],
                                    experiment_id='M2',level=size)
                return eid
            frozen_init = pair('initialization_frozen','pretrained_frozen','random_frozen')
            unfrozen_init = pair('initialization_unfrozen','pretrained_unfrozen','random_unfrozen')
            pretrained_adapt = pair('adaptation_pretrained','pretrained_unfrozen','pretrained_frozen')
            random_adapt = pair('adaptation_random','random_unfrozen','random_frozen')
            for effect,parts,kind in [
                ('initialization',[(frozen_init,.5),(unfrozen_init,.5)],'difference'),
                ('adaptation',[(pretrained_adapt,.5),(random_adapt,.5)],'difference'),
                ('initialization_x_adaptation',[(unfrozen_init,1),(frozen_init,-1)],'interaction'),
                ('initialization_frozen',[(frozen_init,1)],'difference'),
                ('initialization_unfrozen',[(unfrozen_init,1)],'difference'),
                ('adaptation_pretrained',[(pretrained_adapt,1)],'difference'),
                ('adaptation_random',[(random_adapt,1)],'difference')]:
                self.contrast(f'{prefix}:{effect}','AUX_M2',parts,kind,dataset_id=ds,effect=effect,level=size)
            for arm,lr in sorted(arms):
                if arm=='random_unfrozen' and lr!=main_lr:
                    eid = self.estimate(f'{prefix}:random_lr-{lr}',arms[(arm,lr)],arms[(arm,main_lr)],
                                        experiment_id='M2',level=size)
                    self.contrast(eid,'AUX_M2',[(eid,1)],dataset_id=ds,effect='encoder_lr_sensitivity',encoder_lr=lr)

    def m3(self):
        groups = defaultdict(list)
        for row in self.rows('M3'):
            context = row['context']
            groups[(row['dataset_id'],row['model_id'],context['rho'],tuple(context['order']))].append(row)
        for (ds,model,rho,order),rows in groups.items():
            cells = defaultdict(list)
            for row in rows:
                cells[(row['context']['support'],row['context']['risk'])].append(row)
            prefix = f'M3:{ds}:{model}:rho-{rho}:head-'+'-'.join(map(str,order))
            effects = {}
            for support in ['full','tail_reduced']:
                effects['risk_'+support] = self.estimate(f'{prefix}:risk_{support}',
                    cells[(support,'skew')],cells[(support,'uniform')],experiment_id='M3',level=rho,
                    tail_definition='current_head_permutation',tail_label_ids=list(order[-max(1,len(order)//3):]))
            for risk in ['uniform','skew']:
                effects['support_'+risk] = self.estimate(f'{prefix}:support_{risk}',
                    cells[('tail_reduced',risk)],cells[('full',risk)],experiment_id='M3',level=rho,
                    tail_definition='current_head_permutation',tail_label_ids=list(order[-max(1,len(order)//3):]))
            for effect,eid in effects.items():
                self.contrast(eid,'AUX_M3',[(eid,1)],dataset_id=ds,effect=effect,model_id=model)
            self.contrast(f'{prefix}:risk_x_support','AUX_M3',
                [(effects['risk_tail_reduced'],1),(effects['risk_full'],-1)],'interaction',
                dataset_id=ds,effect='risk_x_support',model_id=model)

    def m5(self):
        groups = defaultdict(list)
        for row in self.rows('M5'):
            groups[(row['dataset_id'],row['model_id'],row['condition']['metadata']['size'])].append(row)
        for (ds,model,size),rows in groups.items():
            cells = defaultdict(list)
            for row in rows:
                hp = row['entry']['run']['hparams']
                cap = hp.get('max_features',30000) if model in {'tfidf_lr','tfidf_svm'} else hp.get('vocabulary_limit',30000)
                epoch = hp.get('max_epochs',hp.get('epochs',5 if model=='distilbert' else 10))
                cells[(cap,epoch)].append(row)
            prefix = f'M5:{ds}:{model}:{size}'
            if model=='bilstm':
                capacity = []
                for epoch in [10,20]:
                    eid = self.estimate(f'{prefix}:capacity_epoch-{epoch}',cells[(100000,epoch)],cells[(30000,epoch)],
                                        experiment_id='M5',level=size)
                    capacity.append(eid)
                    self.contrast(eid,'AUX_M5',[(eid,1)],dataset_id=ds,effect='capacity',epoch=epoch,model_id=model)
                for cap in [30000,100000]:
                    eid = self.estimate(f'{prefix}:optimization_cap-{cap}',cells[(cap,20)],cells[(cap,10)],
                                        experiment_id='M5',level=size)
                    self.contrast(eid,'AUX_M5',[(eid,1)],dataset_id=ds,effect='optimization',cap=cap,model_id=model)
                self.contrast(f'{prefix}:capacity_x_optimization','AUX_M5',[(capacity[1],1),(capacity[0],-1)],
                              'interaction',dataset_id=ds,effect='capacity_x_optimization',model_id=model)
            elif model=='distilbert':
                eid = self.estimate(f'{prefix}:optimization',cells[(30000,10)],cells[(30000,5)],experiment_id='M5',level=size)
                self.contrast(eid,'AUX_M5',[(eid,1)],dataset_id=ds,effect='optimization',model_id=model)
            else:
                epoch = next(iter(cells))[1]
                eid = self.estimate(f'{prefix}:capacity',cells[(100000,epoch)],cells[(30000,epoch)],experiment_id='M5',level=size)
                self.contrast(eid,'AUX_M5',[(eid,1)],dataset_id=ds,effect='capacity',model_id=model)

    def interventions(self, pilot_plans):
        plans = pilot_plans.get('discovery_pilots',pilot_plans)
        for node in self.design['discovery_pilot_nodes']:
            pilot_id = node['pilot_id']; plan = plans.get(pilot_id,{})
            if plan.get('status')!='complete':
                raise PermissionError(f'Unresolved real discovery pilot plan: {pilot_id}')
            n = plan.get('n_planned')
            if not isinstance(n,int) or isinstance(n,bool) or not 400<=n<=1200:
                raise ValueError(f'Invalid frozen pilot sample-size plan: {pilot_id}')
            ds = node['dataset_id']; model = 'tfidf_svm' if node['mechanism']=='M1' else 'distilbert'
            anchors = [r for r in self.rows('E1') if r['dataset_id']==ds and r['model_id']==model
                       and r['condition']['metadata']['size']==self.protocol['experiments']['E1']['anchor']
                       and r['entry']['data_seed']==self.protocol['seeds']['data'][0]
                       and r['entry']['training_seed']==(42 if model=='distilbert' else None)]
            ids = {r['entry']['artifact_id'] for r in anchors}
            if len(ids)!=1:
                raise ValueError(f'Fixed anchor checkpoint is unresolved for {pilot_id}')
            checkpoint = next(iter(ids)); seed = self.protocol['seeds']['data'][0]
            view_ids = []
            for arm in ['intervention','control']:
                view_id = f'eval:{pilot_id}:{arm}'
                self.runs[view_id] = {'dataset_id':ds,'model_id':model,'data_seed':seed,
                    'training_seed':42 if model=='distilbert' else None,'evaluation_id':view_id,
                    'checkpoint_run_id':checkpoint,'base_group_namespace':f'paired:{pilot_id}',
                    'n_planned':n,'source_partition':'Q_remaining_groups','pilot_excluded':True,
                    'mechanism':node['mechanism'],'arm':arm}
                view_ids.append(view_id)
            eid = f'E6:{pilot_id}:paired_correctness'
            self.estimates.append({'estimate_id':eid,'dataset_id':ds,'experiment_id':'E6','metric':'correctness',
                'draws':[{'data_seed':seed,'left_runs':[view_ids[0]],'right_runs':[view_ids[1]]}],
                'n_planned':n,'checkpoint_run_id':checkpoint,'pilot_id':pilot_id})
            self.contrast(eid,'F6',[(eid,1)],dataset_id=ds,pilot_id=pilot_id)


def build_contrast_manifest(design: dict[str, Any], run_manifest: dict[str, Any],
                            prepared_metadata: dict[str, Any], protocol: dict[str, Any],
                            pilot_plans: dict[str, Any]) -> dict[str, Any]:
    """Declare all legal main/auxiliary effects without reading any Q outcomes.

    ``pilot_plans`` maps each declared pilot ID to actual completed metadata with
    integer ``n_planned`` (400–1200). F6 arms must later keep common base group IDs
    and sample IDs; each view has its own evaluation_id and the same checkpoint.
    """
    _protocol(design,protocol)
    if run_manifest.get('status')!='resolved' or run_manifest.get('unresolved_nodes'):
        raise PermissionError('Run manifest is unresolved/incomplete')
    if run_manifest.get('design_hash')!=design['design_hash']:
        raise ValueError('Resolved run manifest uses another design')
    planned = [r['run_id'] for r in design['physical_runs']]
    actual = [r['planned_run_id'] for r in run_manifest['runs']]
    if len(actual)!=len(set(actual)) or set(actual)!=set(planned):
        raise ValueError('Resolved physical run coverage is incomplete or duplicated')
    builder = _Contrasts(design,run_manifest,prepared_metadata,protocol)
    builder.curves(); builder.long_tail(); builder.m2(); builder.m3(); builder.m5()
    builder.interventions(pilot_plans)
    datasets = {}
    for ds,meta in _metadata(prepared_metadata).items():
        datasets[ds] = {'task':meta['task'],'label_count':len(meta['label_names']),'partition':'Q'}
        if meta['task'] in {'multilabel','multi_label'}:
            datasets[ds]['ap_mask'] = deepcopy(meta['partitions']['test']['ap_mask'])
            if not datasets[ds]['ap_mask']:
                raise ValueError('Q Macro-AP mask is empty; primary endpoint unavailable')
    result = {'schema_version':1,'status':'resolved','design_hash':design['design_hash'],
              'protocol_hash':design['protocol_hash'],'run_manifest_hash':run_manifest.get('manifest_hash'),
              'datasets':datasets,'runs':builder.runs,'estimates':builder.estimates,
              'contrasts':builder.contrasts,'items':deepcopy(builder.contrasts),
              'inactive_conditions':[dict(deepcopy(c),active=False,status='inactive_infeasible')
                                     for c in design['infeasible_conditions']],
              'estimand':'mean_data_draw(mean_training_seed(per_run_metric_left)-mean_training_seed(per_run_metric_right))',
              'full_pool_draws':1,'probability_ensembling':False,'cross_dataset_metric_averaging':False,
              'official_test_predictions_generated':False}
    result['contrast_manifest_hash'] = canonical_hash(result)
    return result
