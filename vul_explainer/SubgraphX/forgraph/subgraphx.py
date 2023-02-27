import os,sys
import torch,json
from tqdm import tqdm
import time
sys.path.append('/home/DIG-main/dig/xgraph/SubgraphX2')
from models.All_Models import Devign_simplify, config_model, IVDetect_simplify, DeepWukong, RevealModel
from load_dataset import get_dataset, get_dataloader
from forgraph.mcts import MCTS, reward_func
from torch_geometric.data import Batch
from Configures import data_args, mcts_args, reward_args, model_args, train_args
from shapley import GnnNets_GC2value_func, gnn_score
from utils import PlotUtils, find_closest_node_result


def pipeline(max_nodes):
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes

    if data_args.dataset_name == 'mutag':
        data_indices = list(range(len(dataset)))
    else:
        loader = get_dataloader(dataset,
                                batch_size=train_args.batch_size,
                                random_split_flag=data_args.random_split,
                                data_split_ratio=data_args.data_split_ratio,
                                seed=data_args.seed)
        data_indices = loader['test'].dataset.indices
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #gnnNets = Devign_simplify(output_dim = 200, num_steps=6)
    #gnnNets = IVDetect_simplify(output_dim = 200, input_dim = 100)
    gnnNets = RevealModel(input_dim = 100, output_dim = 200)
    #gnnNets = DeepWukong(output_dim=200, input_dim=100)
    gnnNets = gnnNets.to(device)
    checkpoint = torch.load(mcts_args.test_ckpt)
    gnnNets.load_state_dict(checkpoint)
    gnnNets.eval()

    # save_dir = os.path.join('/home/DIG-main/dig/xgraph/SubgraphX2/results',
    #                         f"{mcts_args.dataset_name}_"
    #                         f"{model_args.model_name}_"
    #                         f"{reward_args.reward_method}")
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)

    fidelity_score_list = []
    sparsity_score_list = []
    explain_duration=0.0
    for i in tqdm(data_indices):
        # get data and prediction
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()
        data = dataset[i]
        data_temp=Batch.from_data_list([data.clone()])
        probs = gnnNets(data_temp.x,data_temp.edge_index)
        prediction = probs.squeeze().argmax(-1).item()
        original_score = probs.squeeze()[prediction]

        # get the reward func
        value_func = GnnNets_GC2value_func(gnnNets, target_class=prediction)
        payoff_func = reward_func(reward_args, value_func)

        # find the paths and build the graph
        #result_path = os.path.join(save_dir, f"example_{i}.pt")
        result_path = 'input the save path!!!'+data['name'].split('/')[-1]
        

        # mcts for l_shapely
        mcts_state_map = MCTS(data.x, data.edge_index,
                              score_func=payoff_func,
                              n_rollout=mcts_args.rollout,
                              min_atoms=mcts_args.min_atoms,
                              c_puct=mcts_args.c_puct,
                              expand_atoms=mcts_args.expand_atoms)

        if os.path.isfile(result_path):
            print(data['name'].split('/')[-1]+' has been generated')
            continue
            #results = torch.load(result_path)
        else:
            if data.x.shape[0] < 1000:
                results = mcts_state_map.mcts(verbose=True)
            else:
                print(data['name'].split('/')[-1]+'is too big')
                continue
            #torch.save(results, result_path)

        # l sharply score
        graph_node_x = find_closest_node_result(results, max_nodes=int(data.x.shape[0]*0.2))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        toc = time.perf_counter()
        explain_duration += (toc - tic)
        print(explain_duration)

        with open('input the save path!!!'+data['name'].split('/')[-1],'w') as wp:
            json.dump(graph_node_x.coalition,wp)
        # masked_node_list = [node for node in list(range(graph_node_x.data.x.shape[0]))
        #                     if node not in graph_node_x.coalition]
    #     fidelity_score = original_score - gnn_score(masked_node_list, data, value_func,
    #                                                 subgraph_building_method='zero_filling')
    #     sparsity_score = 1 - len(graph_node_x.coalition) / graph_node_x.ori_graph.number_of_nodes()
    #     fidelity_score_list.append(fidelity_score)
    #     sparsity_score_list.append(sparsity_score)

    #     # visualization
    #     # if hasattr(dataset, 'supplement'):
    #     #     words = dataset.supplement['sentence_tokens'][str(i)]
    #     #     plotutils.plot(graph_node_x.ori_graph, graph_node_x.coalition, words=words,
    #     #                    figname=os.path.join(save_dir, f"example_{i}.png"))
    #     # else:
    #     #     plotutils.plot(graph_node_x.ori_graph, graph_node_x.coalition, x=graph_node_x.data.x,
    #     #                    figname=os.path.join(save_dir, f"example_{i}.png"))

    # fidelity_scores = torch.tensor(fidelity_score_list)
    # sparsity_scores = torch.tensor(sparsity_score_list)
    # return fidelity_scores, sparsity_scores
    return None, 0.8


if __name__ == '__main__':
    fidelity_scores, sparsity_scores = pipeline(15)
    #print(f"Fidelity: {fidelity_scores.mean().item()}, Sparsity: {sparsity_scores.mean().item()}")

