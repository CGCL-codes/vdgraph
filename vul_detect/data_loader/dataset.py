
import os
import copy
import json,glob
import sys

import torch
from dgl import DGLGraph
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from utils import load_default_identifiers, initialize_batch, debug

import random

def read_json(filename):
    #读取文件
    with open(filename.strip(),'r') as f:
        file = json.load(f)
    #文件内容读取到torch.tensor()中
    x = torch.tensor(file['node_features'],dtype=torch.float64)
    num_nodes = x.shape[0]

    edge_index_list = []
    for edge in file['graph']:
        if edge[0] <= num_nodes and edge[2] <= num_nodes:
            edge_index_list.append([edge[0],edge[2]])
    edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()
    
    edge_attr_list = []
    for edge in file['graph']:
        edge_attr_list.append([edge[1]])
    edge_attr = torch.tensor(edge_attr_list)

    #y=[]
    #y.append([file['target']])
    #y=torch.tensor(y)
    y = torch.tensor([file['target']], dtype=int)
    
    data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name = filename.strip().split('/')[-1])
    #torch.save(data,filename+'.pt')
    return data

def read_json2(filename):
    #读取文件
    with open(filename.strip(),'r') as f:
        file = json.load(f)
    #文件内容读取到torch.tensor()中
    x = torch.tensor(file['node_features'],dtype=torch.float64)
    num_nodes = x.shape[0]

    # edge_index_list = []
    # for edge in file['graph']:
    #     if edge[0] <= num_nodes and edge[2] <= num_nodes:
    #         edge_index_list.append([edge[0],edge[2]])
    edge_index_list = file['graph']
    edge_index = torch.tensor(edge_index_list,dtype=torch.long)
    
    # edge_attr_list = []
    # for edge in file['graph']:
    #     edge_attr_list.append([edge[1]])
    # edge_attr = torch.tensor(edge_attr_list)

    #y=[]
    #y.append([file['target']])
    #y=torch.tensor(y)
    y = torch.tensor([file['target']], dtype=int)
    
    data=Data(x=x, edge_index=edge_index, y=y, name = filename.strip().split('/')[-1])
    #torch.save(data,filename+'.pt')
    return data


class Devign(InMemoryDataset):

    def __init__(self, root, num_per_class, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'Devign.pt']

    def process(self):
        data_list = []
        
        # json -> data_list
        #dataset_path=['/home/devign_out/devign_out_ff_novul/','/home/devign_out/devign_out_ff_vul/',
        #           '/home/devign_out/devign_out_qu_novul/','/home/devign_out/devign_out_qu_vul/']
        #for path in dataset_path:
        #dataset_list = glob.glob(path + '*.json')
        with open('/home/nvd_dataset/only_nvd_output/Interpretation.txt','r') as f:##
            dataset_list = f.readlines()
        random.shuffle(dataset_list)
        i = 0
        for data_name in dataset_list:
            data_name = data_name.strip()
            data_name = '/home/nvd_dataset/'+data_name.split('/home/mytest/nvd/')[-1]
            i += 1
            #if i>7:
            #    break
            data = read_json(data_name)
            if(data.num_nodes >= 15):
                data_list.append(data)
            else:
                i -=1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DataSet2:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.read_dataset(test_src, train_src, valid_src)
        dataset = {'train': self.train_examples, 'val': self.valid_examples, 'test':  self.test_examples}
        self.dataset_loader=self.create_dataloader(dataset)
    

    def create_dataloader(self,dataset):
        fobatch=[]
        # for i in range(8):
        #     fobatch.append('n_'+str(i))
        loader = {'train': DataLoader(dataset['train'], batch_size=1, shuffle=True),
                  #'val': DataLoader(dataset['val'], batch_size=data_args.val_bs, shuffle=True),
                  'test': DataLoader(dataset['test'], batch_size=1, shuffle=True),
                  #'explain': DataLoader(dataset['test'], batch_size=data_args.x_bs, shuffle=False)
        }
        return loader


    def read_dataset(self, test_src, train_src, valid_src):
        if train_src is not None:
            print('Reading Train File!',train_src)
            with open(train_src) as fp:
                path_list = fp.readlines()
                i = 0
                for path in tqdm(path_list):
                    #if i>50:
                    #    break
                    path=path.strip()
                    #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                    data_name = path
                    data = read_json(data_name)
                    if(data.num_nodes >= 10):
                        i+=1
                        self.train_examples.append(data)
        random.shuffle(self.train_examples)
        if valid_src is not None:
            print('Reading Valid File!',valid_src)
            with open(valid_src) as fp:
                path_list = fp.readlines()
                i = 0
                for path in tqdm(path_list):
                    #if i>7:
                    #    break
                    path=path.strip()
                    #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                    data_name = path
                    data = read_json(data_name)
                    if(data.num_nodes >= 10):
                        i+=1
                        self.valid_examples.append(data)
        random.shuffle(self.valid_examples)
        record_txt=[]
        if test_src is not None:
            print('Reading Test File!',test_src)
            # with open(test_src) as fp:
            #     path_list = fp.readlines()
            path_list = glob.glob(test_src+'/*.json')
            i = 0
            for path in tqdm(path_list):
                #if i>10:
                #    break
                path=path.strip()
                #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                data_name = path
                data = read_json(data_name)
                if(data.num_nodes >= 10):
                    i+=1
                    self.test_examples.append(data)
                    record_txt.append(path)
            # with open("/home/GNNLRP_model/data/compltet_test.txt", 'w') as p_r:
            #     p_r.writelines(record_txt)
        random.shuffle(self.test_examples)

class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={"features": self.features})
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={"etype": torch.LongTensor([etype_number])})
        

class DataSet:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 100
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, test_src, train_src, valid_src):
        if train_src is not None:
            debug('Reading Train File!',train_src)
            with open(train_src) as fp:
                path_list = fp.readlines()
                for path in tqdm(path_list):
                    with open(path.strip(), 'r') as json_file:
                        pdg = json.load(json_file)
                        if len(pdg[self.n_ident]) == 0 or len(pdg[self.n_ident]) < 9:
                            continue
                        example = DataEntry(datset=self, num_nodes=len(pdg[self.n_ident]), features=pdg[self.n_ident],
                                        edges=pdg[self.g_ident], target=pdg[self.l_ident])
                        if self.feature_size == 0:
                            self.feature_size = example.features.size(1)
                            debug('Feature Size %d' % self.feature_size)
                        self.train_examples.append(example)
                    #if len(self.train_examples) > 7:
                    #    break
            random.shuffle(self.train_examples)

        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                #valid_data = json.load(fp)
                path_list = fp.readlines()
                #for entry in tqdm(valid_data.values()):
                for path in tqdm(path_list):
                    with open(path.strip(), 'r') as json_file:
                        pdg = json.load(json_file)
                    if len(pdg[self.n_ident]) == 0 or len(pdg[self.n_ident]) < 9:
                        continue
                    #if len(entry[self.n_ident]) == 0:
                    #    continue
                    #if len(entry[self.n_ident]) < 9:
                    #    continue
                    #example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                    #                    features=entry[self.n_ident],
                    #                    edges=entry[self.g_ident], target=entry[self.l_ident])
                    example = DataEntry(datset=self, num_nodes=len(pdg[self.n_ident]), features=pdg[self.n_ident],
                                        edges=pdg[self.g_ident], target=pdg[self.l_ident])
                    self.valid_examples.append(example)
                    #if len(self.valid_examples) > 7:
                    #    break
            random.shuffle(self.valid_examples)

        record_txt = []
        debug('Reading Test File!',test_src)
        with open(test_src) as fp:
            path_list = fp.readlines()
            for path in tqdm(path_list):
                with open(path.strip(), 'r') as json_file:
                    pdg = json.load(json_file)
                    if len(pdg[self.n_ident]) == 0 or len(pdg[self.n_ident]) < 9:
                        continue
                    example = DataEntry(datset=self, num_nodes=len(pdg[self.n_ident]), features=pdg[self.n_ident],
                                    edges=pdg[self.g_ident], target=pdg[self.l_ident])
                    self.test_examples.append(example)
                    record_txt.append(path)
        with open("/home/mytest/0day/dataset/0day_test.txt", 'w') as p_r:
            p_r.writelines(record_txt)
        random.shuffle(self.test_examples)

    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)