"""Historically selected, causal forecast corrections at monitoring wells only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .assimilation import historical_offsets, lagged_residual_forecast
from .drift_diag import fair_baselines
from .forcing_experiment import paired_metrics
from .forward import member_datum, nudge_to_observations
from .release import atomic_json, sha256
from .update_experiment import load_context


def select(base, out):
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                    selection_years=[2017,2018,2019,2020], confirmation_years=[2021,2022],
                    retentions=[0.,.25,.5,.75,1.], physical_weights=[0.,.5,1.],
                    trigger='Innovation updates still trail persistence on historical selection.',
                    meaning='Well-level observation forecast only; no physical state or policy claim',
                    selection_rule='minimum matched equal-well RMSE on selection years only')
    path = out/'correction_protocol.json'
    if path.exists():
        raise ValueError('Correction protocol already exists')
    atomic_json(path,protocol)
    _, inp, members, _ = load_context(base,out,torch.device('cpu'))
    hist = np.load(base/'hindcast_00.npz')['heads']
    datum = member_datum(members[0],inp.sids)
    with np.load(out/'historical_updates.npz') as z:
        physical, seasonal, obs, dates = z['none'], z['seasonal'], z['observed'], pd.to_datetime(z['dates'])
        baselines = {k:z[k] for k in ['none','field','previous_month','seasonal']}
    candidates = {}
    configs = {}
    origins = {}
    for year in sorted(set(dates.year)):
        start = int(np.flatnonzero(inp.dates.year==year)[0])
        offset = historical_offsets(inp.obs_h,hist[inp.obs_layer,inp.obs_idx],start,datum)
        state = nudge_to_observations(inp,torch.tensor(hist[...,start-1]),start-1,1.,
                                     taper_km=5.,datum=datum)
        p0 = state[inp.obs_layer,inp.obs_idx].numpy()+offset
        s0 = fair_baselines(inp.obs_h[:,:start],start,0)['clim'][:,-1]
        origins[year] = (p0,s0,inp.obs_h[:,start-1])
    for retention in protocol['retentions']:
        for weight in protocol['physical_weights']:
            name = f'rho{retention:g}_physics{weight:g}'
            result = np.empty_like(obs)
            for year,(p0,s0,o0) in origins.items():
                sel = dates.year==year
                blend = weight*physical[:,sel]+(1-weight)*seasonal[:,sel]
                result[:,sel] = lagged_residual_forecast(blend,obs[:,sel],
                    weight*p0+(1-weight)*s0,o0,retention)
            candidates[name] = result
            configs[name] = dict(retention=retention,physical_weight=weight)
    arrays = {**baselines,**candidates}
    sel = dates.year.isin(protocol['selection_years'])
    training = paired_metrics(obs[:,sel],{k:v[:,sel] for k,v in arrays.items()})
    confirmation = paired_metrics(obs[:,~sel],{k:v[:,~sel] for k,v in arrays.items()})
    best = min(candidates,key=lambda k:training['metrics'][k]['rmse_m'])
    atomic_json(out/'correction_selection.json',dict(selection=training,confirmation=confirmation,
                selected_name=best,selected_config=configs[best],
                protocol_sha256=sha256(out/'correction_protocol.json')))
    print(json.dumps(dict(selected_name=best,selection=training,confirmation=confirmation),indent=2))


def evaluate(base,out):
    _,inp,members,_ = load_context(base,out,torch.device('cpu'))
    selection = json.loads((out/'correction_selection.json').read_text())
    cfg = selection['selected_config']
    with np.load(base/'predictions.npz') as z:
        obs,physical,seasonal,dates = z['observed'],z['observed_pumping'],z['clim'],z['dates']
        frozen_physical = z['climatology']
    origin = np.zeros(len(inp.sids))
    noises = [None,np.random.default_rng(0).normal(0,.5,len(inp.sids))]
    for i,member in enumerate(members):
        hist = np.load(base/f'hindcast_{i:02d}.npz')['heads']
        datum = member_datum(member,inp.sids)
        offset = historical_offsets(inp.obs_h,hist[inp.obs_layer,inp.obs_idx],132,datum)
        for noise in noises:
            state = nudge_to_observations(inp,torch.tensor(hist[...,-1]),131,1.,
                                         taper_km=5.,datum=datum,noise=noise)
            origin += state[inp.obs_layer,inp.obs_idx].numpy()+offset
    origin /= len(members)*len(noises)
    seasonal_origin = fair_baselines(inp.obs_h,132,0)['clim'][:,-1]
    w = cfg['physical_weight']
    prediction = lagged_residual_forecast(w*physical+(1-w)*seasonal,obs,
        w*origin+(1-w)*seasonal_origin,inp.obs_h[:,-1],cfg['retention'])
    seasonal_persistence = lagged_residual_forecast(seasonal,obs,seasonal_origin,inp.obs_h[:,-1])
    previous = np.concatenate([inp.obs_h[:,-1:],obs[:,:-1]],axis=1)
    arrays = dict(selected=prediction,seasonal_persistence=seasonal_persistence,
                  previous_month=previous,physical=physical,seasonal=seasonal)
    arrays['selected_climatology_forcing'] = lagged_residual_forecast(
        w*frozen_physical+(1-w)*seasonal,obs,w*origin+(1-w)*seasonal_origin,
        inp.obs_h[:,-1],cfg['retention'])
    report = paired_metrics(obs,arrays)
    report.update(selected_config=cfg,selection_sha256=sha256(out/'correction_selection.json'),
                  forecast_scope='monitoring wells, next monthly head; no state or policy claim',
                  prospective_validation=False,selection_uses_post2022=False)
    np.savez_compressed(out/'correction_predictions.npz',observed=obs,dates=dates,
                        origin_physical=origin,origin_seasonal=seasonal_origin,**arrays)
    atomic_json(out/'correction_report.json',report)
    print(json.dumps(report,indent=2))


def extended(base,out):
    """Fixed-form one-step replay through August 2026 with climatological future forcing."""
    _,inp,_,_ = load_context(base,out,torch.device('cpu'))
    selection = json.loads((out/'correction_selection.json').read_text())
    cfg = selection['selected_config']
    with np.load(out/'correction_predictions.npz') as z:
        p0,s0 = z['origin_physical'],z['origin_seasonal']
    dates = pd.date_range('2023-01-01','2026-08-01',freq='MS')
    table = pd.read_csv('results/twin/new_data_challenge/predictions.csv',
                        dtype={'station_id':str},parse_dates=['date'])
    obs = table.pivot(index='station_id',columns='date',values='observed_m').reindex(
        index=inp.sids,columns=dates).to_numpy(float)
    with np.load(base/'predictions.npz') as z:
        offset=z['offsets']
    with np.load('results/twin_forward/datum_gate.npz') as z:
        physical=z['heads_mean'][0,inp.obs_layer,inp.obs_idx,132:132+len(dates)]+offset[:,None]
    seasonal=fair_baselines(np.concatenate([inp.obs_h,np.full_like(obs,np.nan)],axis=1),132,0)['clim'][:,132:]
    w=cfg['physical_weight']
    arrays={'hybrid':lagged_residual_forecast(w*physical+(1-w)*seasonal,obs,
              w*p0+(1-w)*s0,inp.obs_h[:,-1],cfg['retention']),
            'seasonal_persistence':lagged_residual_forecast(seasonal,obs,s0,inp.obs_h[:,-1]),
            'previous_month':np.concatenate([inp.obs_h[:,-1:],obs[:,:-1]],axis=1),
            'physical':physical,'seasonal':seasonal}
    report=paired_metrics(obs,arrays)
    report.update(period=['2023-01-01','2026-08-01'],future_forcing='frozen historical climatology',
                  current_observation_used_before_scoring=False,prospective_validation=False,
                  selected_config=cfg,scope='One-step monitoring-well forecasts only')
    report['by_year']={str(year):paired_metrics(obs[:,dates.year==year],
        {k:v[:,dates.year==year] for k,v in arrays.items()},minimum_months=1)
        for year in sorted(set(dates.year))}
    np.savez_compressed(out/'extended_predictions.npz',observed=obs,
                        dates=dates.astype(str).to_numpy(dtype=str),**arrays)
    atomic_json(out/'extended_report.json',report)
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['select','evaluate','extended'])
    ap.add_argument('--base',type=Path,default=Path('results/twin/forcing_experiment'))
    ap.add_argument('--out',type=Path,default=Path('results/twin/update_experiment'))
    args=ap.parse_args()
    globals()[args.stage](args.base,args.out)


if __name__=='__main__':
    main()
