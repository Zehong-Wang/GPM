import time

import yaml
import os.path as osp
import gc
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset

from data.pyg_data_loader import load_data, mol_graphs
from model.model import Model

from task.node import preprocess_node, train_node, eval_node
from task.link import preprocess_link, train_link, eval_link
from task.graph import preprocess_graph, train_graph, eval_graph

from utils.sys import set_memory_limit
from utils.args import get_args
from utils.early_stop import EarlyStopping
from utils.scheduler import get_scheduler
from utils.logger import Logger
from utils.utils import seed_everything, check_path, get_num_params, to_millions

import wandb


def get_preprocess(params):
    task = params["task"]

    if task == "node":
        return preprocess_node
    elif task == "link":
        return preprocess_link
    elif task == "graph":
        return preprocess_graph
    else:
        raise ValueError("Does not support the task in preprocessing.")


def get_train(params):
    task = params["task"]

    if task == "node":
        return train_node
    elif task == "link":
        return train_link
    elif task == "graph":
        return train_graph
    else:
        raise ValueError("Does not support the task in finetuning.")


def get_eval(params):
    task = params["task"]

    if task == "node":
        return eval_node
    elif task == "link":
        return eval_link
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Does not support the task in evaluation.")


def run(params):
    seed_everything(42)  # Make sure the split is the same for each run

    # if params['node_pe'] is not 'none', then the value of 'node_pe_dim' cannot be 0
    assert not (params['node_pe'] != 'none' and params['node_pe_dim'] == 0)
    if params['node_pe'] == 'none':
        params['node_pe_dim'] = 0

    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['device'] = device
    print("Use Device:", device)

    graph, splits = load_data(params)
    if isinstance(graph, Data):
        params['input_dim'] = graph.x.size(1)
        params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    elif isinstance(graph, InMemoryDataset):
        params['input_dim'] = graph._data.x_feat.size(1)
        params['edge_dim'] = graph._data.e_feat.size(1) if graph._data.edge_attr is not None else 0
        if params['dataset'] in mol_graphs:
            params['input_dim'] = 16
            params['edge_dim'] = 16
    elif isinstance(graph, dict):
        params['input_dim'] = graph['train']._data.x_feat.size(1)
        params['edge_dim'] = graph['train']._data.e_feat.size(1) if graph['train']._data.edge_attr is not None else 0
        if params['dataset'] in mol_graphs:
            params['input_dim'] = 16
            params['edge_dim'] = 16

    if params.get('num_tasks') is not None:
        params['output_dim'] = params['num_tasks']
    else:
        params['output_dim'] = graph.y.max().item() + 1

    preprocess = get_preprocess(params)
    train = get_train(params)
    eval = get_eval(params)

    start_time = time.time()
    pattern_set = preprocess(graph, params)
    end_time = time.time()
    print(f"Preprocessing time: {end_time - start_time:.2f}s")

    params['pattern_set'] = pattern_set

    training_time = []
    inference_time = []

    logger = Logger()
    if splits is None:
        splits = range(params['split_repeat'])

    for idx, split in enumerate(splits):
        seed_everything(idx)

        model = Model(params=params).to(device)
        if params['pretrain_data'] != 'none':
            pretrain_path = osp.join(params['save_path'], params['pretrain_data'])
            model.load_state_dict(torch.load(osp.join(pretrain_path, f"epoch_{params['pretrain_epoch']}.pt")))
        if params['linear_probe']:
            model.linear_probe()

        num_params = to_millions(get_num_params(model))
        stopper = EarlyStopping(patience=params["early_stop"])

        if idx == 0:
            print(f'The number of parameters: {num_params}M')

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
            betas=(params['opt_beta1'], params['opt_beta2']), eps=params['opt_eps']
        )
        scheduler = get_scheduler(optimizer, params)

        for epoch in range(1, params['epochs'] + 1):
            start_time = time.time()
            loss = train(graph, model, optimizer, split=split, scheduler=scheduler, params=params)
            end_time = time.time()
            training_time.append(end_time - start_time)

            if epoch % params['eval_every'] == 0 and params['split'] != 'pretrain':
                start_time = time.time()
                result = eval(graph, model, split=split, params=params)
                end_time = time.time()
                inference_time.append(end_time - start_time)

                is_stop = stopper(result)
                logger.log(idx, epoch, loss, result)
                if is_stop:
                    print("Early Stopping at Epoch:", epoch)
                    break

                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                    "training dynamics/train_value": result['train'],
                    "training dynamics/val_value": result['val'],
                    "training dynamics/test_value": result['test'],
                })
            else:
                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                })

            if params['save_every'] != 0 and epoch % params['save_every'] == 0:
                save_path = osp.join(params['save_path'], params['dataset'])
                check_path(save_path)
                torch.save(model.state_dict(), osp.join(save_path, f"epoch_{epoch}.pt"))
                print('Model saved at epoch', epoch)

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best values/train": single_best["train"],
            "best values/val": single_best["val"],
            "best values/test": single_best["test"],
        })

        # After training
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    best = logger.get_best()
    wandb.log({
        "final result/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final result/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final result/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final result/train_mean": best['train']['mean'],
        "final result/val_mean": best['val']['mean'],
        "final result/test_mean": best['test']['mean'],
        "final result/train_std": best['train']['std'],
        "final result/val_std": best['val']['std'],
        "final result/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})
    wandb.log({
        "time/training_mean": np.mean(training_time),
        "time/training_std": np.std(training_time),
        "time/training": "{:.2f} ± {:.2f}".format(np.mean(training_time), np.std(training_time)),
        "time/inference_mean": np.mean(inference_time),
        "time/inference_std": np.std(inference_time),
        "time/inference": "{:.2f} ± {:.2f}".format(np.mean(inference_time), np.std(inference_time))
    })
    wandb.finish()

    # Clear everything
    del graph, pattern_set, logger
    torch.cuda.empty_cache()
    gc.collect()


def main():
    set_memory_limit()  # 90% by default
    params = get_args()

    params['data_path'] = osp.join(osp.dirname(__file__), 'cached_data')
    params['pattern_path'] = osp.join(osp.dirname(__file__), 'cached_patterns')
    params['save_path'] = osp.join(osp.dirname(__file__), 'model')

    data_config = osp.join(osp.dirname(__file__), 'config', 'data.yaml')
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    params['task'] = data_config[params['dataset']]['task']
    params['metric'] = data_config[params['dataset']]['metric']
    params['num_tasks'] = data_config[params['dataset']].get('num_tasks', None)

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), 'config', 'main.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[params['task']][params['dataset']])

    if params['inference']:
        params['epochs'] = 1
        params['eval_every'] = 1
    if params['no_node_pe']:
        params['node_pe'] = 'none'
    if params['no_ap']:
        params['pe_encoder'] = 'none'

    wandb.init(
        project="Next Generation GNN",
        config=params,
        mode="disabled" if params["debug"] else "online"
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
