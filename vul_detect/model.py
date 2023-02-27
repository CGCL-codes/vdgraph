import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import GatedGraphConv,GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import numpy as np

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class xunsqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(dim=1)

class xflat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class GlobalMaxPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        # outputs = x.detach().cpu().tolist()
        # sum_feature = outputs[0]
        # for feature in outputs[1:]:
        #     sum_feature=list(np.add(sum_feature, feature))
        # return torch.tensor(sum_feature,dtype=torch.float32, device=x.device).unsqueeze(0)
        return global_add_pool(x, batch)


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


def de_batchify_graphs(features, batch):
    graphid_to_nodeids = {}
    init_i=0
    split_l=0
    max_bacth=max(batch.tolist())
    for i,index in enumerate(batch.tolist()):
        if index!=init_i:
            graphid_to_nodeids[init_i] = torch.LongTensor(
            list(range(split_l,i))).to(device=features.device)
            init_i+=1
            split_l = i
        if index == max_bacth:
            graphid_to_nodeids[init_i] = torch.LongTensor(
            list(range(i,len(batch.tolist())))).to(device=features.device)
            break
    assert isinstance(features, torch.Tensor)
    vectors = [features.index_select(dim=0, index=graphid_to_nodeids[gid]) for gid in
                range(max_bacth+1)]
    lengths = [f.size(0) for f in vectors]
    max_len = max(lengths)
    for i, v in enumerate(vectors):
        vectors[i] = torch.cat(
            (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                            device=v.device)), dim=0)
    output_vectors = torch.stack(vectors)
    #lengths = torch.LongTensor(lengths).to(device=output_vectors.device)
    return output_vectors



class ExtractFeature(nn.Module):
    def __init__(self,input_dim=200, out_dim=400, num_layers=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5)##0.2
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=input_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) for _ in range(num_layers)])
    def forward(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out


class Devign_simplify(nn.Module):
    def __init__(self, input_dim=100, output_dim=200, num_steps=4):
        super().__init__()
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.classifier = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.relu(self.ggnn(x, edge_index))
        pooled = global_max_pool(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        avg = self.classifier(pooled)
        result = self.sigmoid(avg)
        return result

class IVDetect_simplify(nn.Module):

    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        num_layer = 3
        self.out_dim = output_dim #200
        self.in_dim = input_dim
        self.conv1 = GCNConv(input_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        self.conv3 = GCNConv(output_dim, output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)##0.3
        self.connect = nn.Linear(output_dim, output_dim)
        self.readout = GlobalMaxPool()
        self.__classifier = nn.Linear(output_dim, 2)
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv,edge_index))
        post_conv = self.conv3(post_conv,edge_index)
        pooled = self.readout(post_conv, torch.zeros(post_conv.shape[0], dtype=int, device=post_conv.device))
        y_a = self.__classifier(pooled)
        result = self.softmax(y_a)
        return result




class DevignModel(nn.Module):
    def __init__(self,input_dim, output_dim=200, num_steps=6):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)   # [1,100,4]
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def de_batchify_graphs(self, features=None):
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)

        vectors = [torch.tensor(1)]
        vectors[0] = torch.tensor(features,requires_grad = True)
        output_vectors = torch.stack(vectors)

        return output_vectors

    def get_network_inputs(self, graph, cuda=False, device=None):
        features = graph.ndata['features']
        edge_types = graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return graph, features, edge_types
        pass

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.ggnn(x, edge_index)
        x_i = self.de_batchify_graphs(x)
        h_i = self.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        Y_1 = self.maxpool1(
            self.relu(
                #self.conv_l1(h_i.transpose(1, 2))  # num_node >= 5
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2)) #outputs
                )
            )
        )
        Y_2 = self.maxpool2(
            self.relu(
                #self.conv_l2(Y_1)
                self.batchnorm_1d(
                    self.conv_l2(Y_1) #outputs
                )
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            self.relu(
                #self.conv_l1_for_concat(c_i.transpose(1, 2))
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2)) #ouputs+feature
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            self.relu(
                #self.conv_l2_for_concat(Z_1)
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1) #ouputs+feature
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg)
        return result

class DeepWukong(nn.Module):
    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        self.conv = GCNConv(input_dim, output_dim)
        #hidden_size=2*output_dim
        hidden_size = 256
        layers = [
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)##0.5
        ]
        layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)##0.5
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, 2)
        self.readout = GlobalAddPool()
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Sigmoid()
    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.conv(x, edge_index)
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        hiddens = self.__hidden_layers(pooled)
        avg = self.__classifier(hiddens) 
        result = self.softmax(avg)             
        return result

class IVDetect(nn.Module):
    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        num_layer=3
        self.xunsqueeze = xunsqueeze()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=output_dim, batch_first=True)
        self.gru_combine = nn.GRU(input_size=output_dim, hidden_size=output_dim, bidirectional=True, batch_first=True)
        self.xflat = xflat()
        self.dropout = nn.Dropout(0.3)##0.3
        self.connect = nn.Linear(output_dim*2, output_dim)
        self.convs = nn.ModuleList(
            [
                GCNConv(output_dim, output_dim)
                for _ in range(num_layer - 1)
             ]
        )
        self.relus = nn.ModuleList(
            [
                nn.ReLU(inplace=True)
                for _ in range(num_layer - 1)
            ]
        )
        self.conv3 =GCNConv(output_dim, 2)
        self.readout = GlobalMaxPool()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        x=self.xunsqueeze(x)
        feature_vec, _ = self.gru1(x)
        feature_vec, _ = self.gru_combine(feature_vec)
        feature_vec = self.xflat(feature_vec)
        feature_vec = self.dropout(feature_vec)
        feature_vec = self.connect(feature_vec)
        return feature_vec
        for conv, relu in zip(self.convs, self.relus):
            feature_vec = relu(conv(feature_vec, edge_index))
        conv_output = self.conv3(feature_vec, edge_index)
        pooled = self.readout(conv_output, torch.zeros(conv_output.shape[0], dtype=int, device=conv_output.device))
        result = self.softmax(pooled)
        return result
        
        
class RevealModel(nn.Module):
    def __init__(self,input_dim, output_dim=200, num_steps=6):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.hidden_dim = 400
        self.num_layers = 1
        self.relu = nn.ReLU()
        self.readout = GlobalAddPool()
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.extract_feature=ExtractFeature()
        self.__classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            #nn.Softmax(dim=-1)
            nn.Sigmoid()
        )

    
    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        #outputs = self.relu(self.ggnn(x, edge_index))
        outputs = self.ggnn(x, edge_index)
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        h_a = self.extract_feature(pooled)
        y_a = self.__classifier(h_a)
        return y_a