import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pdb import set_trace as deg
from simclr import SimCLR
from simclr.modules import LogisticRegression
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
from dataset import *
from simclr.modules import VViT
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from datetime import datetime
from model import *
from sklearn import metrics

import pickle

def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    lens = []
    for step, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        # get encoding
        with torch.no_grad():
            h = simclr_model(x, x, hidden_only=True)
        
#         h = h.detach()
        if isinstance(h, list):
            for item in h:
                feature_vector.extend(item.detach().cpu().detach().numpy())
                labels_vector.extend(y.numpy())
            lens.append(len(h))
        else:
            feature_vector.extend(h.detach().cpu().detach().numpy())
            labels_vector.extend(y.numpy())
        

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    selfsupervi_x = []
    lable_y =[]
    supervi = []
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        
        with torch.no_grad():
            h = simclr_model(x, x, hidden_only=True)
        
        xx=[] 
        yy=[]
        if isinstance(h, list):
            for item in h:
                xx.extend(item.detach())
                yy.extend(y)
        else:
            xx.extend(h.detach())
            yy.extend(y)
        
        x,y=torch.stack(xx).cuda(), torch.stack(yy).cuda()
        x0,y0=torch.stack(xx).cpu().detach().numpy(), torch.stack(yy).cpu().detach().numpy()
        selfsupervi_x.extend(x0)
        lable_y.extend(y0)

        output, feature = model(x)
        supervi.extend(feature)
        #print(len(supervi))
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
    #print( len(lable_y), len(selfsupervi_x),len(supervi))
    return loss_epoch, accuracy_epoch, lable_y, selfsupervi_x, supervi 


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    accs, f1,re,pr =[], [],[],[]
    preds, labs = [], []
    for step, (x, y) in enumerate(loader):
        model.zero_grad()
        y = y.to(args.device)

        x = x.to(args.device)

        output, feature = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        y, pred = y.cpu().numpy(), predicted.cpu().numpy()
        preds.extend(pred)
        labs.extend(y)
        accs.append(acc)
        

        loss_epoch += loss.item()
    
    labs = np.array(labs)
    preds= np.array(preds)
#     import pdb
#     pdb.set_trace()
    f1 = f1_score(labs, preds, average='macro')
    re = recall_score(labs, preds, average='macro')
    pr = precision_score(labs, preds, average='macro')
    accs = np.mean(np.array(accs))
    fpr, tpr, thresholds = metrics.roc_curve(labs, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)    
    
    return loss_epoch, accs, pr, re, f1, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config0.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir('logs/'+args.log_name):
        os.mkdir('logs/'+args.log_name)
    
    print('use', args.device)
    
    resum_id = 60
    for resum_id in [ 100]:#range(10,161,10):
        test_X = None
    
        for shot in [1,10 ]:

            if args.dataset == "STL10":
                train_dataset = torchvision.datasets.STL10(
                    args.dataset_dir,
                    split="train",
                    download=True,
                    transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
                test_dataset = torchvision.datasets.STL10(
                    args.dataset_dir,
                    split="test",
                    download=True,
                    transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
            elif args.dataset == "CIFAR10":
                train_dataset = torchvision.datasets.CIFAR10(
                    args.dataset_dir,
                    train=True,
                    download=True,
                    transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
                test_dataset = torchvision.datasets.CIFAR10(
                    args.dataset_dir,
                    train=False,
                    download=True,
                    transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
            elif args.dataset == "FaceForensics":
                train_dataset = dataset_DFD(args,mode='val',filename='../low_shot_setting/%d_shot_train.txt'%shot) 
                test_dataset = dataset_DFD(args,mode='test',filename='../low_shot_setting/10_shot_test.txt') 

            else:
                raise NotImplementedError

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=35,
                shuffle=True,
                drop_last=False,
                num_workers=16,
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=1,
            )
            # datetime object containing current date and time
            now = datetime.now()
            print("now =", now)
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("date and time =", dt_string)


            

            ep_acc = []
            
            if not os.path.isdir('logs/%s/ckpt%d/'%(args.log_name, resum_id)):
                os.mkdir('logs/%s/ckpt%d/'%(args.log_name, resum_id))
            
            logger = open('logs/%s/ckpt%d/%d_shot.txt'% (args.log_name, resum_id, shot), 'a')
            logger.write('#'*100+'\n')
            logger.write('Testing time '+ dt_string + '\n')
            logger.write('#'*100 + '\n')

            print("### Creating features from pre-trained context model ###")
    
#             X, Y = [], []
#             for _ in range(100):
#                 train_X, train_y = inference(train_loader, simclr_model, args.device)
#                 X.extend(train_X)
#                 Y.extend(train_y)
#             X, Y = np.array(X), np.array(Y)

            simclr_model = VViT(args, mode='test')    
            model_fp = os.path.join(args.model_path, "checkpoint_{}.pth".format(resum_id))#args.epoch_num))
            load_model(args, simclr_model, model_path=model_fp)
            simclr_model = simclr_model.to(args.device)
            simclr_model.eval()
            
            if test_X is None:
                test_X, test_y = inference(test_loader, simclr_model, args.device)

#             arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
#                 train_X, train_y, test_X, test_y, args.logistic_batch_size
#             )
            testset = torch.utils.data.TensorDataset(
                torch.from_numpy(test_X), torch.from_numpy(test_y)
            )
            arr_test_loader = torch.utils.data.DataLoader(
                testset, drop_last=False, batch_size=args.logistic_batch_size, shuffle=False
            )
        
            simclr_model = VViT(args, mode='train')    
            model_fp = os.path.join(args.model_path, "checkpoint_{}.pth".format(resum_id))#args.epoch_num))
            load_model(args, simclr_model, model_path=model_fp)
            simclr_model = simclr_model.to(args.device)
            simclr_model.eval()
            
            ## Logistic Regression
            n_classes = 2  # CIFAR-10 / STL-10
            model = LogisticRegression(simclr_model.n_features, n_classes)
            model = model.to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,0.25]).cuda())


            acc1,prs, res, f1s, aucs = [], [], [], [], []
            dict_feature = {}
            for epoch in range(args.logistic_epochs):
                loss_epoch, accuracy_epoch, lable_y, selfsupervi_x, supervi   = train(
                    args, train_loader, simclr_model, model, criterion, optimizer
                )
                #print( len(lable_y), len(selfsupervi_x),len(supervi))
                dict_feature[epoch] = [lable_y, selfsupervi_x, supervi]


                # final testing
                loss_epoch, accuracy_epoch, pr, re, f1, auc = test(
                    args, arr_test_loader, simclr_model, model, criterion, optimizer
                )

                acc1.append(accuracy_epoch)
                prs.append(pr)
                res.append(re)
                f1s.append(f1)
                aucs.append(auc)
                string = '[%d-th checkpoint/eopch: %d]: Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f, AUC: %.4f\n' % (resum_id, epoch, accuracy_epoch, pr, re, f1, auc)
                logger.write(string)
                print(string, end='')

            max_auc = max(aucs)
            max_ind = aucs.index(max_auc)
            acc, pr, re, f1, auc = acc1[max_ind], prs[max_ind], res[max_ind], f1s[max_ind], aucs[max_ind]
            string = '[%d-th checkpoint]: Testing accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f, AUC: %.4f\n' % (resum_id, acc, pr, re, f1, auc)
            logger.write(string)
            print(string, end='')
            logger.close()
            with open('./feature/feature_cp{}_shot{}.pickle'.format(resum_id,shot), 'wb') as f:
                pickle.dump(dict_feature, f)

        

