import argparse
from STELLAR import STELLAR
import numpy as np
import pandas as pd
import scanpy as sc
import os
import json
import torch
import pickle
import anndata
import networkx as nx
from datasets import GraphDataset, prepare_graph

def _plot_umap(adata):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    fig = sc.pl.umap(adata, color=['pred'], size=5, return_fig=True)

    return fig


def _create_results(val_df, pred_prob, pred_prob_list, pred_labels):
    results_df = val_df[['image_name', 'object_id', 'cell_type']].copy()
    results_df['pred'] = pred_labels.tolist()
    results_df['pred_prob'] = pred_prob.tolist()
    results_df['prob_list'] = pred_prob_list.tolist()
    results_df.rename(columns={
        'image_name': 'image_id',
        'object_id': 'cell_id',
        'cell_type': 'label'
    }, inplace=True)

    return results_df


def _create_labels_dict(dataset_df):
    labels = list(set(dataset_df['cell_type']))

    cell_types = np.sort(labels).tolist()
    cell_type_dict = {}
    inverse_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type

    return cell_type_dict, inverse_dict


def _prepare_dataset(train_df, val_df, args, split=None):
    if split is not None:
        processed_graph_file_name = f"dataset_preprocessed_{split}.pkl"
    else:
        processed_graph_file_name = "dataset_preprocessed.pkl"

    processed_graph_file = os.path.join(args.base_path, processed_graph_file_name)
    if args.use_processed_graph and os.path.exists(processed_graph_file):
        print(f'Using preprocessed graph file: {processed_graph_file}')
        packed_graph = pickle.load(open(processed_graph_file, "rb"))
    else:
        print(f'Processing spatial graph from cell coordinates...')
        packed_graph = prepare_graph(train_df, val_df, args.distance_threshold, args.sample_rate)

        # save to .pkl
        with open(processed_graph_file, 'wb') as file:
            pickle.dump(packed_graph, file)

    labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges = packed_graph
    dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)

    if args.compute_graph_statistics:
        avg_node_degree_labeled = _compute_avg_node_degree(labeled_edges)
        avg_node_degree_unlabeled = _compute_avg_node_degree(unlabeled_edges)
        print('Avg Node Degree Labeled = {:.3f} Avg Node Degree Unlabeled = {:.3f}'
              .format(avg_node_degree_labeled, avg_node_degree_unlabeled))

    return dataset


def _compute_avg_node_degree(edge_list):
    graph = nx.Graph(edge_list)
    node_degrees = dict(graph.degree())

    # Compute the average node degree
    total_nodes = len(node_degrees)
    average_node_degree = sum(node_degrees.values()) / total_nodes

    return average_node_degree


def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--base_path', type=str, required=True,
                        help='configuration_path')
    args = parser.parse_args()

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    args.dataset = config['dataset']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.wd = config['wd']
    args.sample_rate = config['sample_rate']
    args.batch_size = config['batch_size']
    args.distance_threshold = config['distance_threshold']
    args.num_heads = config['num_heads']
    args.seed = config['seed']
    args.num_heads = config['num_heads']
    args.num_seed_class = config['num_seed_class']
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.use_processed_graph = config['use_processed_graph']
    args.compute_graph_statistics = config['compute_graph_statistics']

    dataset_df = pd.read_csv(args.dataset, dtype={
        'patient_id': str,
        'image_name': str,
    })
    cell_type_dict, inverse_dict = _create_labels_dict(dataset_df)

    train_df = dataset_df[dataset_df['dataset_type'] == 'training']
    train_df = train_df.drop('dataset_type', axis=1)
    train_df['cell_type'] = train_df['cell_type'].map(cell_type_dict)

    val_df = dataset_df[dataset_df['dataset_type'] == 'test']
    val_df = val_df.drop('dataset_type', axis=1)
    val_df['cell_type'] = val_df['cell_type'].map(cell_type_dict)

    agg_results_df = pd.DataFrame()
    agg_unlabeled_data_x = None
    for split in train_df['split'].unique():
        train_split_df = train_df[train_df['split'] == split]
        train_split_df = train_split_df.drop('split', axis=1)
        val_split_df = val_df[val_df['split'] == split]
        val_split_df = val_split_df.drop('split', axis=1)

        print(f"Running STELLAR on fold {split} ...")

        dataset = _prepare_dataset(train_split_df, val_split_df, args, split=split)

        stellar = STELLAR(args, dataset)
        stellar.train()

        _, pred_prob, pred_prob_list, pred_labels = stellar.pred()

        # reverse map labels to their original keys
        pred_labels = pred_labels.astype('object')
        for i in range(len(pred_labels)):
            if pred_labels[i] in inverse_dict.keys():
                pred_labels[i] = inverse_dict[pred_labels[i]]

        results_df = _create_results(val_split_df, pred_prob, pred_prob_list, pred_labels)
        results_df.to_csv(os.path.join(args.base_path, f'stellar_results_{split}.csv'), index=False)
        agg_results_df = pd.concat([agg_results_df, results_df])

        # plot UMAP of predictions for split
        unlabeled_data_x = dataset.unlabeled_data.x.numpy()
        adata = anndata.AnnData(unlabeled_data_x)
        adata.obs['pred'] = pd.Categorical(results_df['pred'])
        figure = _plot_umap(adata)
        figure.savefig(os.path.join(args.base_path, f'UMAP_predictions_{split}.pdf'), format="pdf", bbox_inches="tight")

        if agg_unlabeled_data_x is None:
            agg_unlabeled_data_x = unlabeled_data_x
        else:
            agg_unlabeled_data_x = np.concatenate([agg_unlabeled_data_x, unlabeled_data_x], axis=0)

    # create results file
    agg_results_df.to_csv(os.path.join(args.base_path, 'stellar_results.csv'), index=False)

    # plot UMAP of predictions
    adata = anndata.AnnData(agg_unlabeled_data_x)
    adata.obs['pred'] = pd.Categorical(agg_results_df['pred'])
    figure = _plot_umap(adata)
    figure.savefig(os.path.join(args.base_path, 'UMAP_predictions.pdf'), format="pdf", bbox_inches="tight")


if __name__ == '__main__':
    main()
