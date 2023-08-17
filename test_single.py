import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook
from dataset import dataset_DFD
from simclr.modules import SingleViT as VViT
import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn import metrics



def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_dataset = dataset_DFD(args,mode = 'test', feature=args.feature) 


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size if args.test_aug!=1 else 1,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 1,
    )

    model = VViT(0,args)

    model = model.to('cuda:0')

    args.current_epoch = 0
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth".format(args.epoch_num))
        model.load_state_dict(torch.load(model_fp)['model_state_dict'])
        print('Pretrained model is loaded on path', model_fp)
        
    model.eval()
    if not args.eval:
        import json
        fp = open('result.json', 'w')
        results = {}
        
    from tqdm import trange
    t = trange(len(test_loader), desc='Accuracy', leave=True)
    with torch.no_grad():
        accset = []
        preds = []
        labs = []
        for _, (x,y) in zip(t, test_loader):
            if args.test_aug==1:

                y = y.to('cuda')
#                 import pdb
#                 pdb.set_trace()
                pred=model(x, forward_mode='test')
                pred=torch.stack(pred)
                pred=torch.argmax(pred,2)
                rs=torch.mode(pred, 0)
                pred=pred[rs[1][0].item()]
#                 pred=torch.mean(pred, 0)
                if args.eval:
                    acc1 = np.mean((pred==y).cpu().numpy())
                    preds.append(pred)
                    labs.append(y.item())
                    accset.append(acc1)
                    t.set_description('Accuracy:%f' % np.mean(np.array(accset)))
                else:
                    pred = torch.argmax(pred,1)
                    results[x['fn']] = 'fake' if pred==1 else 'real'
            else:  

                y=y.to('cuda')
                pred = model(x)
                if args.eval:
                    acc1 = np.mean((torch.argmax(pred,1)==y).cpu().numpy())
                    accset.append(acc1)
                    preds.extend(torch.argmax(pred,1).cpu().numpy())
                    labs.extend(y.cpu().numpy())
                    t.set_description('Accuracy:%f' % np.mean(np.array(accset)))
                else:
                    for p, f in zip(pred, x):
                        f = f['fn']
                        p = torch.argmax(p, 1)
                        
                        results[f] = 'fake' if p==1 else 'real'

        if args.eval:            
            print('-'*80)
#             print('Testing accuracy:', np.mean(np.array(accset)))
            labs = np.array(labs)
            preds= np.array(preds)
    
            f1 = f1_score(labs, preds, average='macro')
            re = recall_score(labs, preds, average='macro')
            pr = precision_score(labs, preds, average='macro')
            accs = np.mean(np.array(accset))
            fpr, tpr, thresholds = metrics.roc_curve(labs, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)    
            string = 'Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f, AUC: %.4f\n' % ( accs, pr, re, f1, auc)
            print(string)
            print('-'*80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config_single_test.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    main(args)
