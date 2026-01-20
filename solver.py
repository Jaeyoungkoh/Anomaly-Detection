import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from utils.utils import *
from eval_methods import *
from model.AnomalyTransformer import AnomalyTransformer
from model.DualTransformer import DualTransformer
from model.VTTPAT import VTTPAT
from model.VTTSAT import VTTSAT
from model.TranAD import TranAD
from model.MTAD_GAT import MTAD_GAT
from model.GDN import GDN
from model.Proposed import Proposed
from model.Proposed_v2 import Proposed_v2
from model.Proposed_v3 import Proposed_v3
# from data_factory.data_loader import *
from data_factory.dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
from tqdm import tqdm

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, args):
        self.patience = args.patience
        self.verbose = True
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = 0
        self.dataset = args.dataset
        self.save_path = args.save_path

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):

    def __init__(self, args):

        # Boundary index 가져오기
        # if args.train_csv == 'train_dmqa6.csv':
        #     args.boundary_index_train = [500, 2700]
        #     args.boundary_index_test = [1000]
        # else:
        #     args.boundary_index_train = None
        #     args.boundary_index_test = None   

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 모델 설정 : 아키텍처에 따라 다른 모델 선택
        model_dict = {
            'AnomalyTransformer' : AnomalyTransformer,
            'MTAD_GAT' : MTAD_GAT,
            'GDN' : GDN,
            'TranAD' : TranAD,
            'VTTPAT' : VTTPAT,
            'VTTSAT' : VTTSAT,
            'DualTransformer' : DualTransformer,
            'Proposed' : Proposed,
            'Proposed_v2' : Proposed_v2,
            'Proposed_v3' : Proposed_v3
        }

        # self.train_loader = get_loader_segment(self.args, mode='train')
        # self.vali_loader = get_loader_segment(self.args, mode='val')
        # self.test_loader = get_loader_segment(self.args, mode='test')

        # load data
        self.train_loader, self.vali_loader, self.test_loader = get_dataloader(self.args)
        # Channel 개수 추출
        first_batch = next(iter(self.train_loader))

        if self.args.model_name in ['GDN', 'Proposed_v2']:
            edge_index_sets = []
            _, self.args.input_c, _ = first_batch[0].shape
            edge_index = first_batch[-1]
            edge_index_sets.append(edge_index[0])
            self.args.edge_index_sets = edge_index_sets
        else:
            _, _, self.args.input_c = first_batch[0].shape


        self.build_model(self.args, model_dict)
        self.criterion = nn.MSELoss()
        self.criterion_tran = nn.MSELoss(reduction='none')
        self.k = args.k
        
        # Early stopping
        self.best_val_total = np.inf                # 아직 최고 성능이 없음을 의미
        self.best_val_total = np.inf                # 아직 최고 성능이 없음을 의미
        self.es_patience    = self.args.patience    # 허용할 최대 미개선 epoch 수
        self.es_counter     = 0                     # 연속 미개선 epoch 수
        self.delta          = 0                     # 최소 개선 폭 (미세 진동 방지용)

        self.log_tensorboard = args.log_tensorboard


        self.args_summary = str(args.__dict__)
        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{self.args.log_dir}")
            self.writer.add_text("args_summary", self.args_summary)

        self.losses = {
            "train_loss": [],
            "val_loss": [],
        }    


    def build_model(self, args, model_dict):
        self.model = model_dict[args.model_name](args)
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)  

        if args.model_name == 'TranAD':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.5) # 5 epoch 마다 0.5(gamma) 감소

        if torch.cuda.is_available():
            self.model.cuda()

    # For early stopping
    def vali(self, vali_loader):
        self.model.eval()

        if self.args.model_name == 'AnomalyTransformer':

            loss_1, loss_2 = [], []

            for i, (input_data, y, label, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach())) + 
                                    torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach(),series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)),series[u].detach())) + 
                                   torch.mean(my_kl_loss(series[u].detach(),(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = self.criterion(output, input)

                loss_1.append((rec_loss - self.args.k * series_loss).item())
                loss_2.append((rec_loss + self.args.k * prior_loss).item())

            return np.average(loss_1), np.average(loss_2)

        elif self.args.model_name in ['MTAD_GAT']:
            
            fore_list = []
            recon_list = []
            loss = []

            for i, (input_data, y, label, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                recons, fore, attns = self.model(input)

                if self.args.model_type == 'reconstruction':
                    rec_loss = self.criterion(recons, input)
                    loss.append(rec_loss.item())

                elif self.args.model_type == 'mix':
                    y = y.float().to(self.device)    
                    y = y.squeeze(1)
                    fore = fore.squeeze(1)

                    fore_loss = self.criterion(fore, y)
                    rec_loss = self.criterion(recons, input)
                    fore_list.append(fore_loss.item())
                    recon_list.append(rec_loss.item())
                    total_loss = fore_loss + rec_loss
                    loss.append(total_loss.item())    

            return np.average(loss)

        elif self.args.model_name in ['GDN']:

            loss = []

            for i, (input_data, y, label, edge_index) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                y = y.float().to(self.device)
                edge_index = edge_index.long().to(self.device)

                output, _, _, _, _ = self.model(input, edge_index)

                rec_loss = self.criterion(output, y)
                loss.append(rec_loss.item())
            
            return np.average(loss)

        elif self.args.model_name in ['Proposed_v2']:

            loss = []

            for i, (input_data, y, label, edge_index) in enumerate(vali_loader): # input_data : (B, C, L)
                input = input_data.float().to(self.device)
                y = y.float().to(self.device)
                edge_index = edge_index.long().to(self.device)

                # Proposed_v2 returns (output, attns)
                output, _ = self.model(input, edge_index)

                rec_loss = self.criterion(output, input)
                loss.append(rec_loss.item())
            
            return np.average(loss)

        elif self.args.model_name in ['TranAD']:

            loss = []

            for i, (input_data, y, label, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                input = input.permute(1,0,2)
                output = self.model(input)

                if isinstance(output, tuple): 
                    output = output[1]

                rec_loss = self.criterion(output, input)
                loss.append(rec_loss.item())

            return np.average(loss)  

        elif self.args.model_name in ['VTTPAT', 'VTTSAT', 'Proposed', 'DualTransformer', 'Proposed_v3']:

            loss = []

            for i, (input_data, y, label, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output, attns = self.model(input)
                rec_loss = self.criterion(output, input)

                loss.append(rec_loss.item())
            
            return np.average(loss)


    def train(self):

        print("======================TRAIN MODE======================")

        path = self.args.save_path

        early_stopping = EarlyStopping(self.args)
        train_steps = len(self.train_loader)

        for epoch in range(self.args.num_epochs):

            loss1_list, loss2_list = [], []
            series_list, prior_list = [], []
            fore_list = []
            recon_list = []
            total_list = []
            epoch_start = time.time()
            self.model.train()



            print(f"Epoch: {epoch} start")
            
            for i, (input_data, y, label, edge_index) in tqdm(enumerate(self.train_loader)):

                self.optimizer.zero_grad()
                input = input_data.float().to(self.device)

                # 매 Batch 마다 실행
                if self.args.model_name == 'AnomalyTransformer':
                    output, series, prior, _ = self.model(input)

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach())) + 
                                        torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach(),series[u])))
                        prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)),series[u].detach())) + 
                                       torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)))))

                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)
                    rec_loss = self.criterion(output, input)

                    loss1 = rec_loss - self.args.k * series_loss
                    loss2 = rec_loss + self.args.k * prior_loss

                    loss1_list.append((rec_loss - self.args.k * series_loss).item())
                    loss2_list.append((rec_loss + self.args.k * prior_loss).item())

                    recon_list.append(rec_loss.item())
                    series_list.append(series_loss.item())
                    prior_list.append(prior_loss.item())

                    # Minimax strategy
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                    self.optimizer.step()

                elif self.args.model_name in ['TranAD']:
                    # input : [B, L, C]    
                    local_bs = input.shape[0]
                    input = input.permute(1,0,2)

                    output = self.model(input)

                    loss1 = self.criterion(output[0], input)
                    loss2 = self.criterion(output[1], input)
                    n = epoch + 1

                    rec_loss = self.criterion(output, input) if not isinstance(output, tuple) else torch.mean((1 / n) * loss1 + (1 - 1/n) * loss2)   

                    recon_list.append(rec_loss.item())

                    rec_loss.backward()
                    self.optimizer.step()

                elif self.args.model_name in ['MTAD_GAT']:
                    recons, fore, attns = self.model(input)

                    if self.args.model_type == 'reconstruction':
                        rec_loss = self.criterion(recons, input)
                        recon_list.append(rec_loss.item())

                        rec_loss.backward()

                    elif self.args.model_type == 'mix':   
                        y = y.float().to(self.device)        
                        fore = fore.squeeze(1)
                        y = y.squeeze(1)

                        fore_loss = self.criterion(fore, y)
                        rec_loss = self.criterion(recons, input)
                        fore_list.append(fore_loss.item())
                        recon_list.append(rec_loss.item())
                        total_loss = fore_loss + rec_loss
                        total_list.append(total_loss.item())
                        total_loss.backward()
                    
                    self.optimizer.step()

                elif self.args.model_name in ['GDN']:
                    y = y.float().to(self.device)
                    edge_index = edge_index.long().to(self.device)

                    output, _, _, _, _ = self.model(input, edge_index)

                    rec_loss = self.criterion(output, y)
                    recon_list.append(rec_loss.item())
                
                    rec_loss.backward()
                    self.optimizer.step()

                elif self.args.model_name in ['Proposed_v2']:
                    y = y.float().to(self.device)
                    edge_index = edge_index.long().to(self.device)

                    # Output: (B, L, C)
                    output, _ = self.model(input, edge_index)

                    rec_loss = self.criterion(output, input)
                    recon_list.append(rec_loss.item())
                
                    rec_loss.backward()
                    self.optimizer.step()

                elif self.args.model_name in ['VTTPAT', 'VTTSAT', 'Proposed', 'DualTransformer', 'Proposed_v3']:

                    output, attns = self.model(input)

                    rec_loss = self.criterion(output, input)
                    recon_list.append(rec_loss.item())

                    rec_loss.backward()
                    self.optimizer.step()

            # 매 Epoch 마다 실행
            if self.args.model_name == 'AnomalyTransformer':         

                train_loss = np.average(loss1_list)
                recon_loss = np.average(recon_list)
                series_list = np.average(series_list)
                prior_list = np.average(prior_list)    

                vali_loss1, vali_loss2 = self.vali(self.vali_loader)
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Recon Loss: {:.7f} Series Loss: {:.7f} Prior Loss: {:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss1, recon_loss, series_list, prior_list))

                early_stopping(vali_loss1, vali_loss2, self.model, path)
                if early_stopping.early_stop:
                     print("Early stopping")
                     break
                
                if self.args.adjust_lr is True:        
                    adjust_learning_rate(self.optimizer, epoch + 1, self.args.lr)


            elif self.args.model_name in ['GDN']:         

                train_loss = np.average(recon_list)    
                vali_loss = self.vali(self.vali_loader)     
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))

                if vali_loss < self.best_val_total:
                    self.best_val_total = vali_loss
                    torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
                    self.es_counter = 0 # Counter reset
                else:
                    self.es_counter += 1                  

                if self.es_counter >= self.es_patience:
                    print(f"\nEarly stopping triggered after"
                          f"{epoch+1 } epochs. Best val-total : {self.best_val_total:.6f}\n")
                    break   

            elif self.args.model_name in ['MTAD_GAT']:         
                
                if self.args.model_type == 'reconstruction':
                    train_loss = np.average(recon_list)    
                    vali_loss = self.vali(self.vali_loader)     
                    print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))

                elif self.args.model_type == 'mix':
                    recon_loss = np.average(recon_list)    
                    fore_loss = np.average(fore_list)
                    train_loss = np.average(total_list)
                    vali_loss = self.vali(self.vali_loader)     
                    print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Recon Loss: {:.7f} Fore Loss: {:.7f} Vali Loss: {:.7f} ".format(epoch + 1, train_steps, train_loss, recon_loss, fore_loss, vali_loss))

                if vali_loss < self.best_val_total:
                    self.best_val_total = vali_loss
                    torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
                    self.es_counter = 0 # Counter reset
                else:
                    self.es_counter += 1                  

                if self.es_counter >= self.es_patience:
                    print(f"\nEarly stopping triggered after"
                          f"{epoch+1 } epochs. Best val-total : {self.best_val_total:.6f}\n")
                    break                     


            elif self.args.model_name in ['TranAD']:         
    
                train_loss = np.average(recon_list)    
                vali_loss = self.vali(self.vali_loader)     
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))

                if vali_loss < self.best_val_total:
                    self.best_val_total = vali_loss
                    torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
                    self.es_counter = 0 # Counter reset
                else:
                    self.es_counter += 1                  

                if self.es_counter >= self.es_patience:
                    print(f"\nEarly stopping triggered after"
                          f"{epoch+1 } epochs. Best val-total : {self.best_val_total:.6f}\n")
                    break     

                self.scheduler.step()

            elif self.args.model_name in ['VTTPAT', 'VTTSAT', 'Proposed', 'DualTransformer', 'Proposed_v2', 'Proposed_v3']:         

                rec_loss = np.average(recon_list)    
                vali_loss = self.vali(self.vali_loader)  

                # Append epoch loss
                self.losses["train_loss"].append(rec_loss)
                self.losses["val_loss"].append(vali_loss)

                if self.log_tensorboard:
                    self.write_loss(epoch)

                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} ".format(epoch + 1, train_steps, rec_loss, vali_loss))

                if vali_loss < self.best_val_total:
                    self.best_val_total = vali_loss
                    torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
                    self.es_counter = 0 # Counter reset
                else:
                    self.es_counter += 1                  

                if self.es_counter >= self.es_patience:
                    print(f"\nEarly stopping triggered after"
                          f"{epoch+1 } epochs. Best val-total : {self.best_val_total:.6f}\n")
                    break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_start))

        if self.vali_loader is None:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))


    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, "model.pt")))
        self.model.eval()
        self.temperature = self.args.temperature

        print("======================TEST MODE======================")
        
        criterion = nn.MSELoss(reduction='none')

        test_labels = []
        actuals = []
        recons = []
        preds = []
        attens_energy = []
        mse_loss = []
        att_scores = []
        temporal_att_storage = {}
        channel_att_storage = {}

        with torch.no_grad():
            for i, (input_data, y, labels, edge_index) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)

                # 매 Batch 마다 실행
                if self.args.model_name == 'AnomalyTransformer':
                    output, series, prior, _ = self.model(input)
                    series_loss = 0.0
                    prior_loss = 0.0                

                    loss = torch.mean(criterion(input, output), dim=-1)

                    for u in range(len(prior)):
                        if u == 0:
                            series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach()) * self.temperature
                            prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)),series[u].detach()) * self.temperature
                        else:
                            series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)).detach()) * self.temperature
                            prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.win_size)),series[u].detach()) * self.temperature

                    metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                    cri = metric * loss
                    cri = cri.detach().cpu().numpy().flatten()
                    mse = loss.detach().cpu().numpy().flatten()

                    actuals.append(input.detach().cpu().numpy())
                    recons.append(output.detach().cpu().numpy())
                    # attens_energy.append(cri)
                    mse_loss.append(cri)
                    test_labels.append(labels.detach().cpu().numpy())

                elif self.args.model_name in ['MTAD_GAT']:   
                    if self.args.model_type == 'reconstruction':
                        recons, fore, attns = self.model(input)
                        loss = torch.mean(criterion(input, recons), dim=-1)  # (B, L)               

                        mse = loss.detach().cpu().numpy().flatten() # (B*L,) 

                        actuals.append(input.detach().cpu().numpy())
                        recons.append(recons.detach().cpu().numpy())
                        att_scores.append(attns.detach().cpu())                
                        mse_loss.append(mse)
                        test_labels.append(labels.detach().cpu().numpy())

                    elif self.args.model_type == 'mix':
                        y = y.float().to(self.device)
                        _, fore_output, _ = self.model(input)

                        recon_input = torch.cat((input[:,1:,:], y.unsqueeze(1)), dim=1)
                        recon_output, _, attns = self.model(recon_input)

                        recon_error = criterion(recon_input, recon_output)
                        recon_loss = torch.mean(recon_error[:,-1,:], dim=-1)
                        fore_loss = torch.mean(criterion(y, fore_output), dim=-1)
                        loss = fore_loss + self.args.gamma * recon_loss

                        mse = loss.detach().cpu().numpy().flatten()

                        actuals.append(y.detach().cpu().numpy())
                        preds.append(fore_output.detach().cpu().numpy())
                        recons.append(recon_output[:,-1,:].detach().cpu().numpy())
                        att_scores.append(attns.detach().cpu())                
                        mse_loss.append(mse)
                        test_labels.append(labels.detach().cpu().numpy())


                elif self.args.model_name in ['GDN']:   
                    y = y.float().to(self.device)
                    edge_index = edge_index.long().to(self.device)

                    output, _, _, _, _= self.model(input, edge_index)

                    loss = torch.mean(criterion(output, y), dim=-1)                

                    mse = loss.detach().cpu().numpy().flatten()

                    actuals.append(y.detach().cpu().numpy())
                    recons.append(output.detach().cpu().numpy())       
                    mse_loss.append(mse)
                    test_labels.append(labels.detach().cpu().numpy())

                elif self.args.model_name in ['TranAD']:   
                    # input : [B, L, C]    
                    input = input.permute(1,0,2)
                    output = self.model(input)

                    if isinstance(output, tuple): 
                        output = output[1]

                    loss = torch.mean(criterion(input, output), dim=-1)  
                    
                    # flatten
                    mse = loss.detach().cpu().numpy().flatten()

                    input = input.permute(1,0,2)
                    output = output.permute(1,0,2)

                    actuals.append(input.detach().cpu().numpy())
                    recons.append(output.detach().cpu().numpy())      
                    mse_loss.append(mse)
                    test_labels.append(labels.detach().cpu().numpy())

                elif self.args.model_name in ['VTTPAT', 'VTTSAT']:   
                    output, _ = self.model(input)

                    loss = torch.mean(criterion(input, output), dim=-1)                

                    mse = loss.detach().cpu().numpy().flatten()

                    actuals.append(input.detach().cpu().numpy())
                    recons.append(output.detach().cpu().numpy())
                    mse_loss.append(mse)
                    test_labels.append(labels.detach().cpu().numpy())

                elif self.args.model_name in ['Proposed', 'DualTransformer', 'Proposed_v2', 'Proposed_v3']:   
                    if self.args.model_name == 'Proposed_v2': 
                        edge_index = edge_index.long().to(self.device)
                        output, attn = self.model(input, edge_index) # B, C, L

                        # shape: (B, C, L) -> Channel 축(dim=1)에 대해 평균 -> (B, L)
                        loss = torch.mean(criterion(input, output), dim=1)       
                        input = input.permute(0, 2, 1)   # (B, C, L) -> (B, L, C)
                        output = output.permute(0, 2, 1) # (B, C, L) -> (B, L, C)                                         
                    else:
                        output, attn = self.model(input) # B, L, C
                        loss = torch.mean(criterion(input, output), dim=-1)  
                                      
                    mse = loss.detach().cpu().numpy()

                    actuals.append(input.detach().cpu().numpy())
                    recons.append(output.detach().cpu().numpy())
                    mse_loss.append(mse)
                    test_labels.append(labels.detach().cpu().numpy())

                    if self.args.output_attention:
                        # 마지막 Layer의 attention score
                        temp_attn_raw = attn['temporal'] # (B,H,L,L)
                        chan_attn_raw = attn['channel'] # (B,C,C)

                        # Multi-Head 평균
                        temp_attn_avg = torch.mean(temp_attn_raw, dim=1) # Shape: (B, L, L)
                        chan_attn_avg = chan_attn_raw # Shape: (B, C, C)

                        bs = input_data.size(0)
                        win_size = self.args.win_size
                        stride = self.args.win_size

                        for j in range(bs):

                            start_idx = i*bs*stride + j*stride
                            end_index = start_idx + win_size - 1

                            # Key: 윈도우의 (시작, 끝) 인덱스 튜플
                            window_key = (start_idx, end_index)

                            # Value: 해당 윈도우의 Attention map (시각화를 위해 numpy로 변환)
                            temporal_att_storage[window_key] = temp_attn_avg[j].detach().cpu().numpy()
                            channel_att_storage[window_key] = chan_attn_avg[j].detach().cpu().numpy()

            if self.args.model_name == 'AnomalyTransformer':

                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1) if attens_energy else attens_energy # 비었을 때 pass
                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)

                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])

            elif self.args.model_name == 'MTAD_GAT':

                attention_tensor = torch.cat(att_scores, dim=0) if att_scores else att_scores
                self.save_tensor(attention_tensor, "attention_scores.pt")

                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)
                
                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])
                if self.args.model_type == 'mix':            
                    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-1]) # (B*L,K)
                    
            elif self.args.model_name == 'GDN':

                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)

                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])

            elif self.args.model_name == 'TranAD':

                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)

                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])

            elif self.args.model_name in ['VTTSAT', 'VTTPAT']:

                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)

                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])

            elif self.args.model_name in ['Proposed', 'DualTransformer', 'Proposed_v2', 'Proposed_v3']:

                mse_loss = np.concatenate(mse_loss, axis=0).reshape(-1)
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_mse = np.array(mse_loss)
                test_labels = np.array(test_labels)

                actuals = np.concatenate(actuals,axis=0).reshape(-1, actuals[0].shape[-1])
                recons = np.concatenate(recons,axis=0).reshape(-1, recons[0].shape[-1])

                if self.args.output_attention:
                    np.save(os.path.join(self.args.save_path, f'temporal_att_storage.npy'), temporal_att_storage)
                    np.save(os.path.join(self.args.save_path, f'channel_att_storage.npy'), channel_att_storage)
                    print("Attention maps saved successfully.")


            ###### Score Section ######
            df_dict = {}
            anomaly_scores = np.zeros_like(actuals)

            for i in range(recons.shape[1]): # Feature 별
                
                if self.args.model_type in ['reconstruction', 'forecasting']:
                    df_dict[f"Recons_{i}"] = recons[:, i] # (N - window, 1)
                    df_dict[f"True_{i}"] = actuals[:, i] # (N - window, 1)
                    recons_error = (recons[:, i] - actuals[:, i]) ** 2
                    a_score = np.sqrt(recons_error)

                elif self.args.model_type == 'mix':
                    df_dict[f"Preds_{i}"] = preds[:, i] # (N - window, 1)
                    df_dict[f"Recons_{i}"] = recons[:, i] # (N - window, 1)
                    df_dict[f"True_{i}"] = actuals[:, i] # (N - window, 1)
                    recons_error = (recons[:, i] - actuals[:, i]) ** 2
                    preds_error = (preds[:, i] - actuals[:, i]) ** 2
                    a_score = np.sqrt(preds_error) + self.args.gamma * np.sqrt(recons_error)
                    

                # IQR 사용 여부 (feature 간 스케일을 맞춰서 비교를 용이하게 만들어주는 역할)
                if self.args.scale_scores:
                    q75, q25 = np.percentile(a_score, [75, 25]) # a_score의 상위 25% 및 하위 25% 지점을 계산
                    iqr = q75 - q25 # iqr : 분산 역할(대안)
                    median = np.median(a_score)
                    a_score = (a_score - median) / (1+iqr)

                # Anomaly Score
                anomaly_scores[:, i] = a_score # (N - window, 1) > (N - window, k)
                df_dict[f"A_Score_{i}"] = a_score

            a_scores_mean = np.mean(anomaly_scores, 1)
            df_dict['A_Score_Global'] = a_scores_mean
            test_df = pd.DataFrame(df_dict)

            # PA%K(=100) AUC
            # scores = test_mse.copy()
            scores = a_scores_mean.copy()
            attack = test_labels.copy()   
            start=np.percentile(scores, 50)
            end=np.percentile(scores, 99)                      
            bf_eval, bf_pred = bf_search(scores, attack, start=start, end=end, step_num=1000, K=100, verbose=False)

            for k, v in bf_eval.items():
                bf_eval[k] = float(v)

            print(f"Results of PA%K=100 AUC using best f1 score search:\n {bf_eval}")

            # bf-method save
            # test_df["A_Score_Global"] = scores
            global_bf_thres = bf_eval["threshold"]
            test_df["A_True_Global"] = attack
            test_df["Thresh_Global_bf"] = global_bf_thres
            test_preds_global = np.array(bf_pred).astype(int) # 1, 0 이진으로 변환
            test_df[f"A_Pred_Global_bf"] = test_preds_global

            count = 0
            count = sum(1 for f in os.listdir(self.args.save_path) if f.startswith("test_output"))
            fname = "test_output.pkl" if count == 0 else f"test_output_{count}.pkl"
            test_df.to_pickle(os.path.join(self.args.save_path, fname))

            # PA%K AUCs using best f1 search method     
            history = dict()
            K = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  

            f1_values = []
            print(f'Threshold start: {np.percentile(scores, 50):.4f} end: {np.percentile(scores, 99):.4f}')

            for k in K:
                bf_eval, _ = bf_search(scores, attack,
                                    start=np.percentile(scores, 50),
                                    end=np.percentile(scores, 99),
                                    step_num=1000,
                                    K=k,
                                    verbose=False)

                f1_values.append(bf_eval['f1'])
                print(f"K: {k}, precision: {bf_eval['precision']:.4f}, recall: {bf_eval['recall']:.4f}, f1: {bf_eval['f1']:.4f}, AUROC: {bf_eval['ROC_AUC']:.4f}, Threshold : {bf_eval['threshold']:.4f}")
                history[f'precision_{k}'] = bf_eval['precision']
                history[f'recall_{k}'] = bf_eval['recall']
                history[f'f1_{k}'] = bf_eval['f1']
                history[f'roc_auc_{k}'] = bf_eval['ROC_AUC']
                history[f'threshold_{k}'] = bf_eval['threshold']

            auc = sum(0.5 * (f1_values[i] + f1_values[i + 1]) * (K[i + 1] - K[i]) for i in range(len(K) - 1)) / 100
            history.setdefault('PA%K AUC', []).append(auc)

            print(f'PA%K AUC: {auc}')

            # Save
            summary_file_name = 'summary.txt'
            with open(f"{self.args.save_path}/f1_{summary_file_name}", "w") as f:
                f.write(f"Threshold Range: {np.percentile(scores, 50):.4f} ~ {np.percentile(scores, 99):.4f}\n\n")
                f.write(f"{'K':<6}{'Precision':<12}{'Recall':<10}{'F1 Score':<10}{'AUROC':<10}{'Threshold':<10}\n")
                f.write("-" * 60 + "\n")

                for i, k in enumerate(K):
                    precision = history[f'precision_{k}']
                    recall = history[f'recall_{k}']
                    f1 = history[f'f1_{k}']
                    roc_auc = history[f'roc_auc_{k}']                    
                    threshold = history[f'threshold_{k}']

                    f.write(f"{k:<6}{precision:<12.4f}{recall:<10.4f}{f1:<10.4f}{roc_auc:<10.4f}{threshold:<10.4f}\n")

                f.write(f"\nPA%K AUC: {auc:.4f}\n")

            print("-- Done.")

            # Save plot
            if self.args.save_auc_curve:
                print("Generating plots...")
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # --- 그래프 1: ROC Curve ---
                # scikit-learn을 사용하여 FPR, TPR 계산
                
                print("Generating publication-quality plots...")
                # 1행 2열의 서브플롯을 생성합니다.
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # 폰트는 Times New Roman을 우선으로 하되, 없을 경우 기본 serif 폰트를 사용합니다.
                try:
                    plt.rcParams['font.family'] = 'Times New Roman'
                except RuntimeError:
                    plt.rcParams['font.family'] = 'serif'

                plt.rcParams['font.size'] = 14  # 기본 폰트 크기
                plt.rcParams['axes.labelsize'] = 16  # 축 레이블 폰트 크기
                plt.rcParams['axes.titlesize'] = 18  # 제목 폰트 크기
                plt.rcParams['xtick.labelsize'] = 12 # x축 눈금 폰트 크기
                plt.rcParams['ytick.labelsize'] = 12 # y축 눈금 폰트 크기
                plt.rcParams['legend.fontsize'] = 14 # 범례 폰트 크기
                plt.rcParams['lines.linewidth'] = 2.5 # 선 굵기
                plt.rcParams['lines.markersize'] = 8 # 마커 크기
                plt.rcParams['figure.dpi'] = 300 # 그림 해상도 (DPI)

                fpr, tpr, _ = roc_curve(attack, scores)
                roc_auc_score_val = roc_auc_score(attack, scores)

                ax1.plot(fpr, tpr, color='crimson', label=f'ROC AUC = {roc_auc_score_val:.4f}')
                ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
                ax1.set_xlim([-0.02, 1.0])
                ax1.set_ylim([0.0, 1.02])
                ax1.set_xlabel('False Positive Rate (FPR)')
                ax1.set_ylabel('True Positive Rate (TPR)')
                ax1.set_title('ROC Curve')
                ax1.legend(loc="lower right")
                ax1.grid(linestyle=':', linewidth=0.5)

                # --- 그래프 2: PA%K Curve (F1-Score vs. K) ---
                ax2.plot(K, f1_values, marker='o', linestyle='-', color='darkblue', label=f'PA%K AUC = {auc:.4f}')
                ax2.fill_between(K, f1_values, color='lightblue', alpha=0.3)
                ax2.set_xlabel('K')
                ax2.set_ylabel('F1-Score')
                ax2.set_title('PA%K Curve')
                ax2.set_ylim([-0.02, 1.02])
                ax2.set_xlim([-2, max(K) + 2])
                ax2.legend(loc="lower right")
                ax2.grid(linestyle=':', linewidth=0.5)


                # 전체 레이아웃을 깔끔하게 조정합니다.
                plt.tight_layout(pad=2.0)

                # --- 그래프 저장 로직 추가 ---
                save_filename = os.path.join(self.args.save_path, "roc_pak_auc_curves.png")
                plt.savefig(save_filename)
                print(f"Plot saved to {save_filename}")


    def save_tensor(self, tensor, file_name):
        PATH = os.path.join(self.args.save_path, file_name)
        if os.path.exists(self.args.save_path):
            pass
        else:
            os.mkdir(self.args.save_path)
        torch.save(tensor, PATH)        \

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)