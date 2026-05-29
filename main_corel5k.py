import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter

import MLdataset
import argparse
import time
from new_model import get_model, AsymmetricBetaLoss
import evaluation
import torch
import numpy as np
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
import math


def loss_singleview_EDL(alpha_c_lb_i, beta_c_lb_i, labels_lb,G,epoch,criterion):

    loss_tensor = criterion(alpha_c_lb_i, beta_c_lb_i, labels_lb)*G
    loss_tensor_row_sums = torch.sum(loss_tensor, dim=1)  
    G_row_sums = torch.sum(G, dim=1) 
    sample_loss = loss_tensor_row_sums / G_row_sums
    return sample_loss

def loss_allview_EDL(alpha_c_lb, beta_c_lb, labels_lb,W,G,epoch,criterion):
   
    loss_sample=[]
    for vi in range(W.shape[1]):
        loss_sample_i=loss_singleview_EDL(alpha_c_lb[vi], beta_c_lb[vi], labels_lb,G,epoch,criterion)
        loss_sample.append(loss_sample_i.unsqueeze(1))
    loss_samples = torch.cat(loss_sample, dim=1)
    loss_samples=loss_samples*W
    loss_tensor_row_sums = torch.sum(loss_samples, dim=1)  
    W_row_sums = torch.sum(W, dim=1)  
    samples_loss = loss_tensor_row_sums / W_row_sums
    total_loss=torch.sum(samples_loss)
   
    return total_loss
        
def relative_diff(e1, e2, eps=1e-8):
    b1 = e1/(e1+e2+2)
    b2 = e2/(e1+e2+2)
    denom =b1 + b2 + eps
    diff = torch.abs(b1 - b2) / denom
    d=(b1+b2)*(1 - diff)
    d = torch.sigmoid(d)
    return d

def train(loader, model, loss_model, opt, sche, epoch,logger,sim_epochs):

    losses = AverageMeter()
    model.train()
    criterion = AsymmetricBetaLoss(clip=args.clip, gamma_neg=args.neg)
    
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
       
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        
      
        
        stacked_alpha,fusion_alpha = model(data,inc_V_ind,inc_L_ind,mode='train')
        
        evidence_alpha_list = [sa[:, 0, :].squeeze(1) for sa in stacked_alpha]
        evidence_beta_list= [sa[:, 1, :].squeeze(1) for sa in stacked_alpha]
        
        loss_EDL = loss_allview_EDL(evidence_alpha_list, evidence_beta_list, label, inc_V_ind,
                                    inc_L_ind, epoch, criterion)
      
        loss_fu = loss_singleview_EDL(fusion_alpha[:, 0, :].squeeze(1), fusion_alpha[:, 1, :].squeeze(1),
                                      label, inc_L_ind, epoch, criterion)
        loss_fu_1 = torch.sum(loss_fu)
        loss_CL = loss_EDL + loss_fu_1 
        
        diss_list=[]
        loss_d = 0
        for i in range(inc_V_ind.shape[1]):
            d = relative_diff(evidence_alpha_list[i], evidence_beta_list[i], 1e-8)  # (n,c)
            loss_d_s = torch.sum(d * inc_V_ind[:, i].unsqueeze(1) * inc_L_ind) / torch.sum(
                inc_V_ind[:, i].unsqueeze(1) * inc_L_ind)
            loss_d = +loss_d_s
            diss_list.append(d)
        diss = relative_diff(fusion_alpha[:, 0, :].squeeze(1) - 1, fusion_alpha[:, 1, :].squeeze(1) - 1, 1e-8)  # (n,c)
        loss_d_f = torch.sum(diss * inc_L_ind) / torch.sum(inc_L_ind)
        loss_d = loss_d_f + loss_d
        

        loss =loss_CL + args.theta * loss_d  
        
        
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()
       
        losses.update(loss.item())
       
        
    
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   
                        losses=losses))
    
    return losses,model

def test(loader, model, loss_model, epoch,logger):
   
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    
    model.eval()

  
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
       
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        
        end = time.time()
        stacked_alpha,fusion_alpha = model(data,inc_V_ind,inc_L_ind,mode='test')#(n,2,c)
        
        pred = fusion_alpha[:,0,:]/(fusion_alpha[:,0,:]+fusion_alpha[:,1,:])
        
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
            
        
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def seed_torch(seed=1029):

	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args,file_path):
    seed_torch(42)
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = False,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        cur_inc_L_ind= torch.tensor(train_dataset.cur_inc_L_ind).float().to('cuda:0')
        
        
        model = get_model(n_stacks=4,n_input=d_list,n_z=args.n_z,Nlabel=classes_num,device=device)
        # print(model)
        loss_model = Loss(0.2, classes_num,  device)

        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        scheduler = None
        

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        
        sim_epochs = []
        
        for epoch in range(args.epochs):
          
            train_losses,model= train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,logger,sim_epochs)

            val_results = test(val_dataloder,model,loss_model,epoch,logger)

            
            if val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25>=static_res:   #adjust weight of each metric
                static_res = val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)
        if len(sim_epochs)>0:
            np.save(f'diction/{args.dataset}_feature.npy',torch.stack(sim_epochs,dim=0).numpy())
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr clip neg best_epoch,batch_size\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.clip),str(args.neg),str(best_epoch),str(args.batch_size)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'final_records'))#'final_records'
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k']) #here to select which dataset you want
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int) # here to set the repeat number
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='10_new_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e0) # not work here
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=500) # here to set the repeat number  
    
    # Training args
    parser.add_argument('--n_z', type=int, default=512) # here to set the dimension
    parser.add_argument('--batch_size', type=int, default=32) # here to set the batch_size
    
    parser.add_argument('--clip', type=float, default=0.2) 
    parser.add_argument("--neg", type=int, default=10)
    
    args = parser.parse_args()
    if args.records_dir:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-1]
    theta_list=[1e0]
    batchsize_list = [96]
    epochs_list = [100]
    clip_list = [0.1]
    neg_list = [10]
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for theta in theta_list:
            args.theta = theta
                             
            for max_epoch in epochs_list:
                args.epochs = max_epoch
                for batch_size in batchsize_list:
                    args.batch_size = batch_size
                    
                    for neg in neg_list:
                        args.neg  = neg
                        for clip in clip_list:
                            args.clip  = clip
                    
                            for dataset in args.datasets:
                                args.dataset = dataset
                                file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                                args.mask_view_ratio) + '_LM_' +
                                                str(args.mask_label_ratio) + '_T_' + 
                                                str(args.training_sample_ratio) + '.txt')
                                args.file_path = file_path
    
                                main(args,file_path)
