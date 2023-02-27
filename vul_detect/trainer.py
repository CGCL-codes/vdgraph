import copy
import imp
import os 
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils import debug

import matplotlib.pyplot as plt
import json
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.long()##
            targets = torch.LongTensor(targets).cuda()
            #targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        #print(all_targets,'\n',all_predictions)
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100, accuracy_score(all_targets, all_predictions) * 100,recall_score(all_targets, all_predictions) * 100, precision_score(all_targets, all_predictions)
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        prediction_rec = dict()
        i = 0
        with open('/home/mytest/0day/dataset/0day_test.txt','r') as f:
            all_files = f.readlines()
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.long()##
            targets = torch.LongTensor(targets).cuda()##
            #targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            # prediction_rec[all_files[i]] = predictions.numpy().tolist()
            cnt = 0
            if i+8 < len(all_files):
                for file in all_files[i: i+8]:
                    prediction_rec[file] = predictions.numpy().tolist()[cnt]
                    cnt += 1
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            i += 8
            print("------> ",i)
        # model.train()
        #print(all_targets,'\n',all_predictions)
        with open('/home/mytest/0day/dataset/rec_0day_test2.txt','w') as f_j:
            json.dump(prediction_rec, f_j)

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=20):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    best_acc = 0
    f11 = []
    accc = []
    step = []
    t_loss = []
    v_loss = []
    try:
        for step_count in range(max_steps):
            for index, data in tqdm(enumerate(dataset['train'])):
                model.train()
                model.zero_grad()
                # x = data.x
                # targets = data.y
                # edge_index=data.edge_index
                # x=x.cuda()
                # edge_index=edge_index.cuda()
                graph, targets = dataset.get_next_train_batch()
                #print(targets)
                #targets = targets.reshape(8,1)
                #targets = targets.float()##
                targets = targets.long()##
                #targets = torch.zeros(8,2).scatter_(1, targets, 1)
                #targets = torch.FloatTensor(targets).cuda()##
                targets = torch.LongTensor(targets).cuda()
                predictions = model(graph,cuda=True) #开始forward
                #print(predictions)
                # with torch.no_grad():
                    # predictions = model(graph, cuda=True)
                #print(predictions.shape,targets.shape)
                batch_loss = loss_function(predictions, targets)
                if log_every is not None and (step_count % log_every == log_every - 1):
                    debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
                train_losses.append(batch_loss.detach().cpu().item())
                #print('LOSS:\t',train_losses)
                batch_loss.backward()
                optimizer.step()
                if step_count % dev_every == (dev_every - 1):
                    valid_loss, valid_f1 ,valid_acc, valid_recall ,valid_precision = evaluate_loss(model, loss_function, dataset.initialize_test_batch(),
                                                        dataset.get_next_test_batch)
                    step.append(step_count)
                    t_loss.append(batch_loss.detach().cpu().item())
                    v_loss.append(valid_loss)
                    f11.append(valid_f1)
                    accc.append(valid_acc)
                    if valid_acc > 68.0 and valid_f1 > 72:
                        ckptname = 'nvd-' + str(dataset.batch_size) +'-' + str(valid_f1) + '-' + str(valid_acc) +'-'+str(valid_recall)+'-DevignModel_2d.ckpt'
                        _save_ckpt_file = open('/home/mytest/nvd_novul/model/' + ckptname, 'wb')
                        torch.save(model.state_dict(), _save_ckpt_file)
                        _save_ckpt_file.close()
                    # unuse
                    if valid_f1 > 50.0 and valid_f1 > best_f1:
                        patience_counter = 0
                        best_f1 = valid_f1
                        #best_acc = valid_acc
                        best_model = copy.deepcopy(model.state_dict())
                        #_save_file = open(save_path + '-model.bin', 'wb')
                        #torch.save(model.state_dict(), _save_file)
                        #_save_file.close()
                    else:
                        patience_counter += 1
                    debug('Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tf1: %5.2f\tacc: %5.2f\trecall:%5.2f\tprecision:%5.2f\tPatience %d' % (
                        step_count, np.mean(train_losses).item(), valid_loss, valid_f1, valid_acc, valid_recall, valid_precision, patience_counter))
                    debug('=' * 100)
                    train_losses = []
                    if patience_counter == max_patience:
                        break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')
    if best_model is not None:
        model.load_state_dict(best_model)
    #_save_file = open(save_path + '-model.bin', 'wb')
    #torch.save(model.state_dict(), _save_file)#_save_file.close()
    _save_ckpt_file = open(save_path + '-model.ckpt', 'wb')
    torch.save(model.state_dict(), _save_ckpt_file)
    _save_ckpt_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1 :%0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)

    plt.subplot(2, 2, 1)
    plt.plot(step, t_loss)
    plt.title('Train_Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(step, v_loss)
    plt.title('Valid_Loss')

    plt.subplot(2, 2, 3)
    plt.plot(step, f11)
    plt.title('F1')

    plt.subplot(2, 2, 4)
    plt.plot(step, accc)
    plt.title('Acc')

    plt.suptitle('Evaluate Metrics')
    plt.savefig('evaluate_metrics.jpg')

def eval(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    _save_ckpt_file = torch.load(save_path + 'nvd-8-74.98554077501447-71.29107202124129-DevignModel_2d.ckpt')
    #_save_ckpt_file = torch.load('/home/mytest/nvd_novul/model/nvd-DevignModel_newedge-model.ckpt')
    # model.load_state_dict(_save_ckpt_file['state_dict'])
    model.load_state_dict(_save_ckpt_file)
    #_save_ckpt_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)
