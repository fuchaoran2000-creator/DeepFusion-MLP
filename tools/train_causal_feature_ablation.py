#!/usr/bin/python3
"""Controlled same-budget pose-feature ablation on captured causal streams."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mentor_benchmark import (GROUPS, features, train_model, predict, corrected,
                             metrics, load_sequence, validate_manifest, choose_alpha, dump)

CONFIGS = ['full20', 'mask_P_O', 'mask_P_C', 'mask_D_O', 'mask_D_C', 'mask_E',
           'mask_increments', 'poses8', 'motion_difference12', 'relative6_pose', 'relative32_pose']


def subset(seq, config):
    if config.startswith('relative'):
        x = features(seq, 'relative32')
        return x[:, :6] if config == 'relative6_pose' else x
    x = features(seq, 'full20').copy()
    if config.startswith('mask_'):
        groups = ['D_O', 'D_C'] if config == 'mask_increments' else [config[5:]]
        for group in groups:
            x[:, GROUPS[group]] = 0
    elif config == 'poses8':
        x = x[:, :8]
    elif config == 'motion_difference12':
        x = x[:, 8:]
    elif config != 'full20':
        raise ValueError(config)
    return x


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    dump(args.output/'manifest.json', manifest)
    plan = dict(configs=CONFIGS, seeds=[11, 22, 33], target='pose',
                max_epochs=args.epochs, patience=25, batch_size=256,
                primary_ablation='Zero masked groups before training normalization; all seven main variants retain 20 inputs and 7683 parameters.',
                compact_controls='8/12/6/32 actual inputs; every variant has the same pose-residual target.',
                input_contract='causal_grid_v1 actual captured input pairs, shared missing-history reset',
                model_selection='Early stopping and alpha on validation only; no confidence gate in this feature study.',
                test_history='Already inspected public development sequences; not a fresh final holdout.')
    plan_path=args.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Existing feature-study plan differs; use a new output directory.')
    dump(plan_path, plan)
    rows = {r['sequence_id']: r for r in manifest['sequences']}
    train = [load_sequence(rows[n]) for n in manifest['train']]
    val = [load_sequence(rows[n]) for n in manifest['validation']]
    frozen = []
    for config in CONFIGS:
        for seed in plan['seeds']:
            directory = args.output/f'{config}_seed{seed}'
            directory.mkdir(exist_ok=True)
            path = directory/'model.pt'
            if not path.exists():
                model, norm, training = train_model(train, val, config, 'pose', seed, args.epochs, feature_builder=subset)
                outputs = [predict(model, norm, s, config, subset)[0] for s in val]
                alpha, score = choose_alpha(val, outputs, 'pose')
                torch.save(dict(model_state_dict=model.state_dict(), norm=norm,
                                config=config, seed=seed, target='pose', alpha=alpha,
                                validation_ape_m=score, training=training), path)
                dump(directory/'training.json', training)
                print(f'Frozen {config}/{seed}: validation APE={score:.6f}, alpha={alpha:.2f}', flush=True)
            frozen.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(args.output/'selection_freeze.json', dict(models=frozen, test_reference_not_loaded_yet=True))
    # Evaluation references first enter this study after all model choices freeze.
    test = [load_sequence(rows[n]) for n in manifest['test']]
    results = []
    for config in CONFIGS:
        for seed in plan['seeds']:
            state = torch.load(args.output/f'{config}_seed{seed}/model.pt', map_location='cpu', weights_only=False)
            from residual_models import Regressor
            model = Regressor(len(state['norm']['xm']), 'mlp', False)
            model.load_state_dict(state['model_state_dict']); model.eval()
            for seq in test:
                mean = predict(model, state['norm'], seq, config, subset)[0]
                pose = corrected(seq, mean, state['alpha'], 'pose')
                raw, stats = metrics(seq, seq.cart), metrics(seq, pose)
                results.append(dict(config=config, seed=seed, sequence=seq.name, scene=seq.scene,
                    input_dim=len(state['norm']['xm']), parameters=state['training']['parameter_count'],
                    alpha=state['alpha'], raw_ape_m=raw['ape_m']['rmse'], ape_m=stats['ape_m']['rmse'],
                    raw_rpe_1s_m=raw['rpe']['1s']['translation_m']['rmse'],
                    rpe_1s_m=stats['rpe']['1s']['translation_m']['rmse'],
                    input_available_fraction=seq.coverage['input_available_fraction'],
                    reference_valid_fraction=seq.coverage['reference_valid_fraction']))
    with (args.output/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    dump(args.output/'provenance.json', dict(
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ['tools/train_causal_feature_ablation.py', 'mentor_benchmark.py', 'causal_stream.py']},
        model_count=len(frozen), result_rows=len(results)))
    print('Completed causal feature study', args.output, len(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=150)
    run(p.parse_args())
