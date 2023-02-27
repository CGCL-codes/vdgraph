import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import GatedGraphConv,GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


def config_model(model, args):
    model.to(args.device_id)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    print(f'Loading best checkpoint ... ')

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class GlobalMaxPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

class Devign_simplify(nn.Module):
    def __init__(self, output_dim=200, num_steps=6):
        super().__init__()
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.readout = GlobalMaxPool()
        self.classifier = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.relu(self.ggnn(x, edge_index))
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        avg = self.classifier(pooled)
        result = self.sigmoid(avg)
        return result, x

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
        self.dropout = nn.Dropout(0.3)##0.3
        self.connect = nn.Linear(output_dim, output_dim)
        self.readout = GlobalMaxPool()
        self.__classifier = nn.Linear(output_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv,edge_index))
        post_conv = self.conv3(post_conv,edge_index)
        pooled = self.readout(post_conv, torch.zeros(post_conv.shape[0], dtype=int, device=post_conv.device))
        #pooled = self.readout(post_conv, batch)
        y_a = self.__classifier(pooled)
        result = self.softmax(y_a)
        return result, x

class DeepWukong(nn.Module):
    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        self.conv = GCNConv(input_dim, output_dim)
        hidden_size=2*output_dim
        #hidden_size = 256
        layers = [
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)##0.5
        ]
        layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, 2)
        self.readout = GlobalAddPool()
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.Sigmoid()
    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.conv(x, edge_index)
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        hiddens = self.__hidden_layers(pooled)
        avg = self.__classifier(hiddens) 
        result = self.softmax(avg)             
        return result, x

class ExtractFeature(nn.Module):
    def __init__(self,input_dim=200, out_dim=400, num_layers=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2)##0.2
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=input_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) for _ in range(num_layers)])
    def forward(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

class RevealModel(nn.Module):
    def __init__(self,input_dim=100, output_dim=200, num_steps=6):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.hidden_dim = 400
        self.num_timesteps = num_steps
        self.num_layers = 1
        self.relu = nn.ReLU()
        self.readout = GlobalAddPool()
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.extract_feature=ExtractFeature()
        # self.__classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # self.softmax=nn.Softmax(dim=1)
        self.__classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            nn.Softmax(dim=-1)
            #nn.Sigmoid()
        )

    
    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.relu(self.ggnn(x, edge_index))
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        h_a = self.extract_feature(pooled)
        y_a = self.__classifier(h_a)
        #result = self.softmax(y_a)
        return y_a, x
