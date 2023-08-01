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
    results_df = val_df[['sample_id', 'object_id', 'cell_type']].copy()
    results_df['pred'] = pred_labels.tolist()
    results_df['pred_prob'] = pred_prob.tolist()
    results_df['prob_list'] = pred_prob_list.tolist()
    results_df.rename(columns={
        'sample_id': 'image_id',
        'object_id': 'cell_id',
        'cell_type': 'label'
    }, inplace=True)

    return results_df


def _create_labels_dict(train_df, val_df):
    train_labels = list(set(train_df['cell_type']))
    val_labels = list(set(val_df['cell_type']))
    labels = list(set(train_labels + val_labels))

    cell_types = np.sort(labels).tolist()
    cell_type_dict = {}
    inverse_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type

    return cell_type_dict, inverse_dict


def _prepare_dataset(train_df, val_df, args, compute_graph_statistics=True):
    processed_graph_file = os.path.join(args.base_path, "dataset_preprocessed.pkl")
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

    if compute_graph_statistics:
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

    args.train_dataset = config['train_dataset']
    args.val_dataset = config['val_dataset']
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

    train_df = pd.read_csv(args.train_dataset)
    train_df = train_df[train_df['cell_type'] != 'unlabeled'].reset_index(drop=True)

    val_df = pd.read_csv(args.val_dataset)
    val_df = val_df[val_df['cell_type'] != 'unlabeled'].reset_index(drop=True)

    cell_type_dict, inverse_dict = _create_labels_dict(train_df, val_df)

    train_df['cell_type'] = train_df['cell_type'].map(cell_type_dict)
    dataset = _prepare_dataset(train_df, val_df, args)
    
    stellar = STELLAR(args, dataset)
    stellar.train()

    _, pred_prob, pred_prob_list, pred_labels = stellar.pred()

    # reverse map labels to their original keys
    pred_labels = pred_labels.astype('object')
    for i in range(len(pred_labels)):
        if pred_labels[i] in inverse_dict.keys():
            pred_labels[i] = inverse_dict[pred_labels[i]]

    # create results file
    results_df = _create_results(val_df, pred_prob, pred_prob_list, pred_labels)
    results_df.to_csv(os.path.join(args.base_path, 'stellar_results.csv'), index=False)

    # plot UMAP of predictions
    adata = anndata.AnnData(dataset.unlabeled_data.x.numpy())
    adata.obs['pred'] = pd.Categorical(pred_labels)
    figure = _plot_umap(adata)
    figure.savefig(os.path.join(args.base_path, 'UMAP_predictions.pdf'), format="pdf", bbox_inches="tight")


if __name__ == '__main__':
    main()
