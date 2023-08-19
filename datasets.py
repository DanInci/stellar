import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd

def get_hubmap_edge_index(pos, regions, distance_thres):
    # construct edge indexes when there is region information
    edge_list = []
    regions_unique = np.unique(regions)
    for reg in regions_unique:
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        dists = pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for (i, j) in region_edge_list:
            edge_list.append([locs[i], locs[j]])
    return edge_list


def get_own_edge_index(df, distance_threshold):
    edge_list = []

    # we have to independently construct the graph for each sample
    # map local cell id to a global cell id

    start_cell_index = 0
    samples = df['sample_id'].unique()
    for sample_id in samples:
        sample_df = df[df['sample_id'] == sample_id]
        pos = sample_df[['x', 'y']].values
        dists = pairwise_distances(pos)
        dists_mask = dists < distance_threshold
        np.fill_diagonal(dists_mask, 0)
        sample_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()

        # map sample cell ids in order to be globally unique
        for (i, j) in sample_edge_list:
            new_i = start_cell_index + i
            new_j = start_cell_index + j
            edge_list.append([new_i, new_j])

        start_cell_index += sample_df.shape[0]  # don't forget to increment the offset required for global mapping

    return edge_list

def get_tonsilbe_edge_index(pos, distance_thres):
    # construct edge indexes in one region
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list

def load_hubmap_data(labeled_file, unlabeled_file, distance_thres, sample_rate):
    train_df = pd.read_csv(labeled_file)
    test_df = pd.read_csv(unlabeled_file)
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    test_df = test_df.sample(n=round(sample_rate*len(test_df)), random_state=1)
    train_df = train_df.loc[np.logical_and(train_df['tissue'] == 'CL', train_df['donor'] == 'B004')]
    test_df = test_df.loc[np.logical_and(test_df['tissue'] == 'CL', test_df['donor'] == 'B005')]
    train_X = train_df.iloc[:, 1:49].values # node features, indexes depend on specific datasets
    test_X = test_df.iloc[:, 1:49].values
    labeled_pos = train_df.iloc[:, -6:-4].values # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = test_df.iloc[:, -5:-3].values
    labeled_regions = train_df['unique_region']
    unlabeled_regions = test_df['unique_region']
    train_y = train_df['cell_type_A'] # class information
    cell_types = np.sort(list(set(train_df['cell_type_A'].values))).tolist()
    # we here map class in texts to categorical numbers and also save an inverse_dict to map the numbers back to texts
    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_edges = get_hubmap_edge_index(labeled_pos, labeled_regions, distance_thres)
    unlabeled_edges = get_hubmap_edge_index(unlabeled_pos, unlabeled_regions, distance_thres)
    return train_X, train_y, test_X, labeled_edges, unlabeled_edges, inverse_dict


def prepare_graph(train_df, test_df, distance_threshold, sample_rate):
    train_df = train_df.sample(n=round(sample_rate * len(train_df)), random_state=1)
    train_X = train_df.iloc[:, 9:].values
    test_X = test_df.iloc[:, 9:].values
    train_y = train_df['cell_type'].values

    labeled_edges = get_own_edge_index(train_df, distance_threshold)
    unlabeled_edges = get_own_edge_index(test_df, distance_threshold)

    return train_X, train_y, test_X, labeled_edges, unlabeled_edges

def load_tonsilbe_data(filename, distance_thres, sample_rate):
    df = pd.read_csv(filename)
    train_df = df.loc[df['sample_name'] == 'tonsil']
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    test_df = df.loc[df['sample_name'] == 'Barretts Esophagus']
    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['cell_type'].str.lower()
    labeled_pos = train_df.iloc[:, -4:-2].values
    unlabeled_pos = test_df.iloc[:, -4:-2].values
    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_edges = get_tonsilbe_edge_index(labeled_pos, distance_thres)
    unlabeled_edges = get_tonsilbe_edge_index(unlabeled_pos, distance_thres)
    return train_X, train_y, test_X, labeled_edges, unlabeled_edges, inverse_dict

class GraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, transform=None,):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T, y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data