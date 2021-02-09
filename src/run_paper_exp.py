import os
import sys
import time
import shlex
import subprocess
import argparse
import numpy as np
from itertools import product


def run(cmds, cuda_id):
    _cur = 0

    def recycle_devices():
        for cid in cuda_id:
            if cuda_id[cid] is not None:
                proc = cuda_id[cid]
                if proc.poll() is not None:
                    cuda_id[cid] = None

    def available_device_id():
        for cid in cuda_id:
            if cuda_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cid

        args = shlex.split(cmd)
        exp_dir = args[-1]
        os.makedirs(exp_dir, exist_ok=True)
        log = open('{}/log.txt'.format(exp_dir), 'w')
        print(time.asctime(), ' '.join(args))

        proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)

        cuda_id[cid] = proc

    while _cur < len(cmds):
        recycle_devices()
        cid = available_device_id()

        if cid is not None:
            print(f'CUDA {cid} available for job ({_cur+1} of {len(cmds)})')
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Neural-Network Experiments')
    parser.add_argument('--forecast_date', type=str, required=True)
    # parser.add_argument('--level', type=str, default='county')
    # parser.add_argument('--horizons', type=str, default='7,14,21,28')
    # parser.add_argument('--targets', type=str, default='confirmed,deaths')
    parser.add_argument('--horizons', type=str, default='7')
    parser.add_argument('--targets', type=str, default='confirmed')
    # parser.add_argument('--model_type', type=str, default='zgcn')
    parser.add_argument('--seed_range', type=str, default='0-30')
    parser.add_argument('--cuda', type=str, default='0,1,2,3')

    args = parser.parse_args()
    fdate = args.forecast_date
    horizons = [h for h in args.horizons.split(',')]
    targets = [t for t in args.targets.split(',')]
    # model_type = args.model_type
    seed_start, seed_end = (int(s) for s in args.seed_range.split('-'))
    cuda_id = dict([(str(i), None) for i in args.cuda.split(',')])

    cmds = []
    for target in ['confirmed', 'deaths']:
    # for target in ['confirmed']:
        for horizon in [7,]:
            for seed in range(seed_start, seed_end):
                use_lr = True
                use_mobility = False
                use_fea_zscore = False
                use_adapt_norm = False
                prepro_type = 'none'
                # topo_loss_weight = 0.1
                saint_batch_size = 500
                data_fp = f'../data/daily_us_{horizon}.csv'
                graph_fp = '../data/us_graph.cpt'
                lookback_days = 28
                val_days = 1
                # model_type = 'zgcn'
                rnn_type = 'nbeats'
                gcn_type = 'gcn'
                saint_shuffle_order = 'node_first'
                gcn_aggr = 'max'
                topo_loss_node_num = -1
                pair_loss_node_num = -1
                pair_loss_weight = 0
                for (use_popu_norm, use_logy), (model_type, use_default_edge, topo_loss_weight) in product(
                    [
                        (True, False),
                    ],
                    [
                        ('zgcn', False, 0.01),
                        ('zgcn', False, 0.0),
                        ('zgcn', True, 0.01),
                        ('nbeats', False, 0.0)
                    ],
                ):
                    exp_dir = f'../PaperExp-Main_us_{fdate}/{target}_{horizon}_{model_type}_seed{seed}_pn{use_popu_norm}_logy{use_logy}_de{use_default_edge}_tlw{topo_loss_weight}'
                    if os.path.exists(exp_dir):
                        print(f'Already created {exp_dir}, ...skip...')
                        continue
                    items = [
                        'python graph_task.py',
                        f'--use_mobility {use_mobility}',
                        f'--use_lr {use_lr}',
                        f'--use_popu_norm {use_popu_norm}',
                        f'--use_logy {use_logy}',
                        f'--use_fea_zscore {use_fea_zscore}',
                        f'--use_adapt_norm {use_adapt_norm}',
                        '--early_stop_epochs 20',
                        '--use_saintdataset true',
                        f'--prepro_type {prepro_type}',
                        f'--random_seed {seed}',
                        f'--data_fp {data_fp}',
                        f'--graph_fp {graph_fp}',
                        f'--forecast_date {fdate}',
                        f'--label {target}_target',
                        f'--horizon {horizon}',
                        f'--lookback_days {lookback_days}',
                        f'--val_days {val_days}',
                        f'--saint_batch_size {saint_batch_size}',
                        f'--saint_shuffle_order {saint_shuffle_order}',
                        f'--model_type {model_type}',
                        f'--rnn_type {rnn_type}',
                        f'--gcn_type {gcn_type}',
                        f'--gcn_aggr {gcn_aggr}',
                        f'--use_default_edge {use_default_edge}',
                        f'--topo_loss_node_num {topo_loss_node_num}',
                        f'--topo_loss_weight {topo_loss_weight}',
                        f'--pair_loss_node_num {pair_loss_node_num}',
                        f'--pair_loss_weight {pair_loss_weight}',
                        f'--exp_dir {exp_dir}',
                    ]
                    cmd = ' '.join(items)
                    # we must set exp_dir at last to set proper log file paths
                    cmds.append(cmd)

    run(cmds, cuda_id)
