"""Historical selection, fixed evaluation, and transient sensitivity of head updates.

This conditional model-development experiment is not prospective validation. The
physics parameters already saw 2012–2022; new updater settings use historical years
only. The post-2022 screen is kept separate from candidate selection.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .assimilation import HeadObservationOperator, historical_offsets
from .drift_diag import fair_baselines
from .forcing_experiment import paired_metrics
from .forward import build_model, load_members, member_datum, nudge_to_observations, rollout
from .release import atomic_json, sha256

VARIANTS = {'base': {}, 'storage025': {'S': .25}, 'storage05': {'S': .5},
            'storage2': {'S': 2.}, 'storage4': {'S': 4.},
            'transmissivity05': {'T': .5}, 'transmissivity2': {'T': 2.},
            'leakage025': {'L': .25}, 'leakage4': {'L': 4.}}


def register(base, out):
    original = json.loads((base / 'protocol.json').read_text())
    p = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
             kind='retrospective-update-and-transient-diagnostic',
             selection_years=[2017, 2018, 2019, 2020], confirmation_years=[2021, 2022],
             evaluation_period=['2023-01-01', '2025-07-01'],
             parameter_files=original['parameter_files'],
             input_cache_sha256=sha256(base / 'inputs.local.pkl'),
             forcing_sha256=sha256(base / 'forcing.npz'),
             observations_sha256=sha256(base / 'predictions.npz'),
             gain_candidates=[.25, .5, 1.], radius_km_candidates=[2., 5., 10.],
             transient_candidates=VARIANTS,
             selection_rule='minimum equal-well RMSE on selection years; freeze before evaluation',
             forcing='recorded pumping; recharge climatology fitted before each origin',
             offsets='fixed per-well residual mean strictly before each origin; no future fitting',
             updates='store monthly prediction before updating with current observations',
             limitations=[
                 'Physics parameters saw 2012–2022: historical screens are conditional algorithm selection.',
                 'Post-2022 data were examined previously: retrospective, not prospective validation.',
                 'Pumping is inferred from electricity and known retrospectively.',
                 'Assimilation changes storage; increments are not physical fluxes.',
                 'No subsidence or policy acceptance.'])
    path = out / 'protocol.json'
    if path.exists():
        raise ValueError('Protocol already registered; preserve the existing experiment')
    atomic_json(path, p)


def load_context(base, out, device):
    p = json.loads((out / 'protocol.json').read_text())
    for name, key in [('inputs.local.pkl', 'input_cache_sha256'),
                      ('forcing.npz', 'forcing_sha256'), ('predictions.npz', 'observations_sha256')]:
        if sha256(base / name) != p[key]:
            raise ValueError(f'Frozen input changed: {name}')
    # Only the locally produced cache whose hash was registered above is deserialized.
    with (base / 'inputs.local.pkl').open('rb') as f:
        inp = pickle.load(f)
    paths = [Path('results/twin_runs/stage3_datum_gate') / name for name in p['parameter_files']]
    for path in paths:
        if sha256(path) != p['parameter_files'][path.name]:
            raise ValueError('Frozen model parameters changed')
    members = load_members([str(path) for path in paths])
    op = HeadObservationOperator.from_inputs(inp, device)
    return p, inp, members, op


def prepare_model(inp, member, device, variant):
    model, scalars, _ = build_model(inp.grid, member, device)
    if 'delay_Sd' in scalars or 'aqt_Sa' in scalars:
        raise ValueError('This experiment does not implement updates of slow stores')
    with torch.no_grad():
        for key, factor in variant.items():
            getattr(model, 'log_' + key).add_(np.log(factor))
        if variant.get('S', 1.0) > 1.0:
            model.log_S.clamp_(max=np.log(.3))
    return model, scalars


def run_sequence(inp, model, scalars, state, energy, recharge, dates, observations,
                 offset, learned_datum, config, op, h0):
    """Return priors in observation space plus state-increment magnitudes."""
    device = state.device
    energy, recharge = torch.as_tensor(energy).to(state), torch.as_tensor(recharge).to(state)
    ground = inp.ground_elev.to(state)
    f_inp = replace(inp, obs_h=observations, obs_h_filled=observations,
                    dates=dates, ic_month0_filled=False)
    predictions, increments = [], []
    offset_t = torch.as_tensor(offset, dtype=state.dtype, device=device)
    for t, date in enumerate(dates):
        prior = rollout(model, scalars, state, energy[:, t:t+1], recharge[:, t:t+1], ground,
                         month0=date.month-1, apex_from=h0)[..., -1]
        predictions.append(op.observe(prior, offset_t).detach().cpu().numpy())
        method = config['method']
        if method == 'none':
            state = prior
        elif method == 'innovation':
            state = op.update(prior, observations[:, t], offset_t,
                              config['gain'], config['radius_km'])
        else:
            datum = offset if method == 'field_consistent' else learned_datum
            state = nudge_to_observations(f_inp, prior, t, 1., taper_km=5., datum=datum)
        increments.append(float(torch.sqrt(torch.mean((state-prior)**2)).cpu()))
    return np.stack(predictions, axis=-1), increments


def historical_screen(inp, member, op, device, configs, years, variant=None):
    model, scalars = prepare_model(inp, member, device, variant or {})
    h0 = inp.initial_heads(0).to(device)
    hist = rollout(model, scalars, h0, inp.E_total[:, 1:].to(device),
                   inp.recharge_field[:, 1:].to(device), inp.ground_elev.to(device), apex_from=h0)
    physical = hist[inp.obs_layer, inp.obs_idx].cpu().numpy()
    datum = member_datum(member, inp.sids)
    predictions = {key: [] for key in configs}
    predictions.update(seasonal=[], previous_month=[])
    obs_blocks, date_blocks = [], []
    for year in years:
        start = int(np.flatnonzero(inp.dates.year == year)[0])
        stop = start + 12
        offset = historical_offsets(inp.obs_h, physical, start, datum)
        state = nudge_to_observations(inp, hist[..., start-1], start-1, 1.,
                                     taper_km=5., datum=datum)
        energy = inp.E_total[:, start:stop]
        # Future weather is unavailable at issuance: use prefix-only climatology.
        recharge = torch.stack([inp.recharge_field[:, :start][:, inp.dates[:start].month == m].mean(1)
                                for m in range(1, 13)], dim=1)
        obs = inp.obs_h[:, start:stop]
        for name, config in configs.items():
            values, _ = run_sequence(inp, model, scalars, state, energy, recharge,
                                      inp.dates[start:stop], obs, offset, datum, config, op, h0)
            predictions[name].append(values)
        predictions['seasonal'].append(fair_baselines(inp.obs_h[:, :stop], start, 0)['clim'][:, start:stop])
        predictions['previous_month'].append(inp.obs_h[:, start-1:stop-1])
        obs_blocks.append(obs)
        date_blocks.extend(inp.dates[start:stop].astype(str))
        print(f'Historical block {year} completed ({len(configs)} arms)', flush=True)
    arrays = {key: np.concatenate(value, axis=1) for key, value in predictions.items()}
    observations = np.concatenate(obs_blocks, axis=1)
    return paired_metrics(observations, arrays), arrays, observations, np.array(date_blocks)


def select(base, out, device):
    p, inp, members, op = load_context(base, out, device)
    configs = {'none': {'method': 'none'}, 'field': {'method': 'field'},
               'field_consistent': {'method': 'field_consistent'}}
    for gain in p['gain_candidates']:
        for radius in p['radius_km_candidates']:
            configs[f'innovation_g{gain:g}_r{radius:g}'] = dict(
                method='innovation', gain=gain, radius_km=radius)
    years = p['selection_years'] + p['confirmation_years']
    _, arrays, obs, dates = historical_screen(inp, members[0], op, device, configs, years)
    selected = pd.to_datetime(dates).year.isin(p['selection_years'])
    training = paired_metrics(obs[:, selected], {k:v[:, selected] for k,v in arrays.items()})
    confirmation = paired_metrics(obs[:, ~selected], {k:v[:, ~selected] for k,v in arrays.items()})
    name = min((k for k in configs if k.startswith('innovation')),
               key=lambda k: training['metrics'][k]['rmse_m'])
    report = dict(protocol_sha256=sha256(out/'protocol.json'), selection=training,
                  confirmation=confirmation, selected_name=name, selected_config=configs[name],
                  selection_uses_post2022=False)
    np.savez_compressed(out/'historical_updates.npz', observed=obs, dates=dates, **arrays)
    atomic_json(out/'selection.json', report)
    print('Selected', name, json.dumps(report), flush=True)


def dynamics(base, out, device):
    p, inp, members, op = load_context(base, out, device)
    selected = json.loads((out/'selection.json').read_text())['selected_config']
    years = p['selection_years'] + p['confirmation_years']
    all_arrays, obs, dates = {}, None, None
    for name, variant in p['transient_candidates'].items():
        _, arrays, obs, dates = historical_screen(inp, members[0], op, device,
                                                  {'none': {'method':'none'}, 'innovation':selected},
                                                  years, variant)
        for key, values in arrays.items():
            all_arrays[name+'_'+key] = values
        np.savez_compressed(out/f'historical_{name}.npz', observed=obs, dates=dates, **arrays)
        print('Transient candidate finished', name, flush=True)
    mask = pd.to_datetime(dates).year.isin(p['selection_years'])
    training = paired_metrics(obs[:, mask], {k:v[:, mask] for k,v in all_arrays.items()})
    confirmation = paired_metrics(obs[:, ~mask], {k:v[:, ~mask] for k,v in all_arrays.items()})
    names = list(p['transient_candidates'])
    best = min(names, key=lambda name:training['metrics'][name+'_innovation']['rmse_m'])
    best_free = min(names, key=lambda name:training['metrics'][name+'_none']['rmse_m'])
    atomic_json(out/'dynamics.json', dict(selection=training, confirmation=confirmation,
                                         selected_variant=best, selected_free_variant=best_free,
                                         variants=p['transient_candidates']))
    print('Selected transient variant', best, 'free-run', best_free, flush=True)


def evaluate(base, out, device):
    p, inp, members, op = load_context(base, out, device)
    config = json.loads((out/'selection.json').read_text())['selected_config']
    dynamics_path = (out/'dynamics_extended.json' if (out/'dynamics_extended.json').exists()
                     else out/'dynamics.json')
    dynamics = json.loads(dynamics_path.read_text())
    variant = dynamics['variants'][dynamics['selected_variant']]
    free_variant = dynamics['variants'][dynamics['selected_free_variant']]
    from .forward import future_forcing
    from .scenario import BASELINE

    with np.load(base/'predictions.npz') as z:
        obs, dates = z['observed'], pd.to_datetime(z['dates'])
        old = {key:z[key] for key in ['observed_pumping','clim']}
    with np.load(base/'forcing.npz') as z:
        energy = z['energy']
    _, recharge, _ = future_forcing(inp, BASELINE, len(dates))
    noises = [None, np.random.default_rng(0).normal(0, .5, len(inp.sids))]
    configs = {'field': {'method':'field'}, 'field_consistent': {'method':'field_consistent'},
               'innovation': config, 'innovation_transient':config, 'free_transient':{'method':'none'}}
    # Freeze the selected configurations before evaluating the newer record.
    atomic_json(out/'evaluation_config.json', dict(config=config, transient=variant,
                free_transient=free_variant, selection_sha256=sha256(out/'selection.json'),
                dynamics_sha256=sha256(dynamics_path), protocol_sha256=sha256(out/'protocol.json')))
    sums = {key:np.zeros_like(obs) for key in configs}
    increments = {key:[] for key in configs}
    h0 = inp.initial_heads(0).to(device)
    groups = {}
    for key in configs:
        changes = (variant if key == 'innovation_transient' else
                   free_variant if key == 'free_transient' else {})
        groups.setdefault(json.dumps(changes, sort_keys=True), []).append(key)
    replay_audit = json.loads((base/'hindcast_audit.json').read_text())
    for mi, member in enumerate(members):
        datum = member_datum(member, inp.sids)
        for encoded, keys in groups.items():
            changes = json.loads(encoded)
            model, scalars = prepare_model(inp, member, device, changes)
            if not changes:
                path = base/f'hindcast_{mi:02d}.npz'
                if sha256(path) != replay_audit['files'][path.name]:
                    raise ValueError('Historical replay cache changed')
                hist = torch.tensor(np.load(path)['heads'],device=device)
            else:
                hist = rollout(model, scalars, h0, inp.E_total[:,1:].to(device),
                               inp.recharge_field[:,1:].to(device), inp.ground_elev.to(device), apex_from=h0)
            physical = hist[inp.obs_layer, inp.obs_idx].cpu().numpy()
            offset = historical_offsets(inp.obs_h, physical, len(inp.dates), datum)
            for ni, noise in enumerate(noises):
                state = nudge_to_observations(inp, hist[...,-1], len(inp.dates)-1, 1.,
                                             taper_km=5., datum=datum, noise=noise)
                for key in keys:
                    values, inc = run_sequence(inp, model, scalars, state, energy, recharge,
                                               dates, obs, offset, datum, configs[key], op, h0)
                    sums[key] += values
                    increments[key].extend(inc)
                    np.savez_compressed(out/f'eval_{mi:02d}_{ni}_{key}.npz', prediction=values)
        print('Evaluation member',mi+1,'/',len(members),flush=True)
    arrays = {key:value/(len(members)*len(noises)) for key,value in sums.items()}
    arrays.update(no_update=old['observed_pumping'], seasonal=old['clim'],
                  previous_month=np.concatenate([inp.obs_h[:,-1:],obs[:,:-1]],axis=1))
    baseline_full = fair_baselines(np.concatenate([inp.obs_h, np.full_like(obs, np.nan)], axis=1),
                                   len(inp.dates), 0)['clim']
    previous_seasonal = baseline_full[:, len(inp.dates)-1:-1]
    arrays['seasonal_persistence'] = old['clim'] + arrays['previous_month'] - previous_seasonal
    report = paired_metrics(obs, arrays)
    report.update(protocol_sha256=sha256(out/'protocol.json'), selected_config=config,
                  selected_transient=dynamics['selected_variant'],
                  selected_free_transient=dynamics['selected_free_variant'],
                  members=len(members)*len(noises), prospective_validation=False,
                  state_increment_grid_rms_m={k:float(np.mean(v)) for k,v in increments.items()},
                  limitations=p['limitations'], dates=[str(dates[0].date()),str(dates[-1].date())])
    report['by_year'] = {str(year):paired_metrics(obs[:,dates.year==year],
                          {k:v[:,dates.year==year] for k,v in arrays.items()},minimum_months=1)
                         for year in sorted(set(dates.year))}
    report['by_layer'] = {str(layer+1):paired_metrics(obs[inp.obs_layer==layer],
                          {k:v[inp.obs_layer==layer] for k,v in arrays.items()}) for layer in range(4)}
    np.savez_compressed(out/'predictions.npz', observed=obs, dates=dates.astype(str).to_numpy(dtype=str),
                        sids=np.array(inp.sids), layers=inp.obs_layer, **arrays)
    atomic_json(out/'report.json', report)
    print(json.dumps(report,indent=2),flush=True)


def extend(base, out, device):
    """Bounded historical extension when the optimum hits the storage-screen boundary."""
    p, inp, members, op = load_context(base, out, device)
    original = json.loads((out/'dynamics.json').read_text())
    if original['selected_variant'] != 'storage4':
        raise ValueError('Storage extension is only justified by the upper-bound optimum')
    config = json.loads((out/'selection.json').read_text())['selected_config']
    candidates = {f'storage{s}': {'S':float(s)} for s in (8,16,32)}
    path = out/'dynamics_extension_protocol.json'
    if path.exists():
        raise ValueError('Extension already registered')
    atomic_json(path, dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                reason='Historical optimum hit the initial storage upper bound.',
                variants=candidates, maximum_storativity=.3,
                selection_uses_post2022=False,
                interpretation='Sensitivity of assimilation memory; not a calibrated physical storage estimate'))
    report = original
    for name, variant in candidates.items():
        model, _ = prepare_model(inp,members[0],device,variant)
        if float(model.log_S.exp().max()) > .3:
            raise ValueError('Storage candidate exceeds registered exploratory ceiling')
        _, arrays, obs, dates = historical_screen(inp,members[0],op,device,
            {'none':{'method':'none'},'innovation':config},
            p['selection_years']+p['confirmation_years'],variant)
        np.savez_compressed(out/f'historical_{name}.npz',observed=obs,dates=dates,**arrays)
        report['variants'][name] = variant
        print('Extension candidate completed',name,flush=True)
    all_arrays = {}
    for name in report['variants']:
        with np.load(out/f'historical_{name}.npz') as z:
            obs, dates = z['observed'],pd.to_datetime(z['dates'])
            all_arrays.update({name+'_'+k:z[k] for k in ['none','innovation','previous_month','seasonal']})
    mask=dates.year.isin(p['selection_years'])
    report['selection']=paired_metrics(obs[:,mask],{k:v[:,mask] for k,v in all_arrays.items()})
    report['confirmation']=paired_metrics(obs[:,~mask],{k:v[:,~mask] for k,v in all_arrays.items()})
    for key,arm in [('selected_variant','innovation'),('selected_free_variant','none')]:
        report[key]=min(report['variants'],key=lambda name:report['selection']['metrics'][name+'_'+arm]['rmse_m'])
    report['extension_protocol_sha256']=sha256(path)
    atomic_json(out/'dynamics_extended.json',report)
    print('Frozen extended selections',report['selected_variant'],report['selected_free_variant'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['register','select','dynamics','extend','evaluate'])
    ap.add_argument('--base',type=Path,default=Path('results/twin/forcing_experiment'))
    ap.add_argument('--out',type=Path,default=Path('results/twin/update_experiment'))
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    if args.stage=='register':
        register(args.base,args.out)
    else:
        from .calibrate_flow import set_compile_matvec
        set_compile_matvec(True)
        globals()[args.stage](args.base,args.out,torch.device(args.device))


if __name__=='__main__':
    main()
