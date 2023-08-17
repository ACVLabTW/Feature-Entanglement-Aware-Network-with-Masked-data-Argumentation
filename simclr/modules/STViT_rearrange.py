import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import timm
from vit.vit_pytorch.vit_pytorch import ViT
import numpy as np
from einops.layers.torch import Rearrange
from simclr.modules.gmlp import gMLPVision as mlp
from efficientnet_pytorch import EfficientNet
from pdb import set_trace as sts

import seaborn as sns
import matplotlib.pyplot as plt

# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import to_undirected
# from pygcn.models import GraphConvolution as GCNConv

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        ngf = 512
        nga = 256
        ngfa = 128
        nc = 3
        nz = 1024
        self.args = args
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf , 5, 3, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf , nga , 5, 2, 0, bias=False),
            nn.BatchNorm2d(nga ),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(nga , ngfa , 7, 2, 1, bias=False),
            nn.BatchNorm2d(ngfa),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngfa , nc, 7, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )


    def forward(self, x):        
        # print('ori', x.shape)
        x = self.main(x)
        # print('gan',x.shape)
        x = torch.nn.functional.interpolate(x, size=(self.args.crop_size, self.args.crop_size), mode='bilinear', align_corners=False)
        # print('interpolate', x.shape)
        return x
    
class SingleViT2(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, device, args, mode='train'):
        super(SingleViT2, self).__init__()
        
        
        self.mode=mode
        self.FREQ=args.FREQ
        self.backbone=args.backbone
        self.device = device
        self.n_features = args.n_features
        self.crop_size = args.crop_size
        self.image_size = args.image_size
        self.FRR=args.FRR
        self.FREQ=args.FREQ
        self.heads = args.heads
        self.L1_depth=args.L1_depth
        # self.random_mask = args.random_mask
        self.masked_reconstruction = args.masked_reconstruction
        self.gen = Generator(args) if self.masked_reconstruction else None
        self.use_mask = args.use_mask
        self.percent = args.percent
        print(f"Mask the spatiotemporal features of the background={self.use_mask}")
        print(f"Masked_reconstruction={self.masked_reconstruction}")

        in_feat_dim = 2048
        if 'resnet50' in self.backbone:
            model = torchvision.models.resnet50(pretrained=True) #featMap: -2
        elif 'resnet101' in self.backbone:
            model = torchvision.models.resnet101(pretrained=True)
        elif 'cspdarknet53' in self.backbone:
            in_feat_dim = 1024
            model = timm.create_model('cspdarknet53', pretrained=True) # featmap: -1
        elif 'efficientnet' in self.backbone:
            in_feat_dim = 1792
            model = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            print('Backbone fail!!')
            raise
        
        if 'efficientnet' not in self.backbone and self.backbone != 'dct':
            print('extract the last-%d-th layer' % -args.useFeatMap)

            model = torch.nn.Sequential(*list(model.children())[:args.useFeatMap])
        
        self.encoder = model  #output shape [B, 1024, 8, 8]，DCT=[B,784, 8,8] (ori=224*224)..
        self.reshape =  Rearrange('b c h w -> b (h w) (c)')
        self.recover =  Rearrange('b (h w) c -> b c h w', h=5, w=5)

        if 'efficient' in self.backbone:
            shapes = model.extract_features(torch.zeros((1,3,self.crop_size, self.crop_size))).shape
            print(f'Feature map {self.backbone} size is', shapes)
        else:
            shapes = model(torch.zeros((1,3,self.crop_size, self.crop_size))).shape
            print(f'Feature map {self.backbone} size is', shapes)
        # num_patches = shapes[-1]*shapes[-2]*args.FRR
        num_patches = shapes[-3] # to fit covariance matrix
        
        print(f'in_feat_dim={in_feat_dim} and num_patches={num_patches}')
        
        if args.heads==0:
            self.spatial_transformer = mlp(dim = in_feat_dim,
                                           num_patches=num_patches, # Feat map size
                                           depth = args.L1_depth
                                          )
                                  
        else:
            self.spatial_transformer  = ViT(dim = in_feat_dim,
                        depth = args.L1_depth,
                        heads = args.heads,
                        num_patches=num_patches,
                        mlp_dim = args.n_features,
                        dropout = 0.1,
                        emb_dropout = 0.1)

        self.in_feat_dim = in_feat_dim
        self.patch_size = shapes[-1]*shapes[-2]
        
        self.inter_feat = nn.Sequential(
            nn.Linear(in_feat_dim, args.n_features, bias=False),
            nn.LeakyReLU(inplace=True),
            )
        
        self.cls = nn.Sequential( nn.Linear(args.n_features, 2))
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        
    def stforward(self, X, return_seq=True):
        lstm_i, module_outputs = [], []
        
        if len(X['img'])==self.FRR:
            for k in range(len(X['img'])):
                if 'efficientnet' in self.backbone:
                    # x = self.encoder.extract_features(X['img'][k].to(self.device, non_blocking=True))
                    x = self.encoder.extract_features(X['mask'][k].to(self.device, non_blocking=True))

                elif self.encoder is not None:
                    
                    # x = self.encoder(X['img'][k].to(self.device, non_blocking=True))
                    x = self.encoder(X['mask'][k].to(self.device, non_blocking=True))

                else:
                    # x = X['img'][k].float().to(self.device, non_blocking=True)
                    x = X['mask'][k].float().to(self.device, non_blocking=True)
                # print('before reshape',x.shape)


                # print('X mask', X['mask'][k].shape)
                # print('conv feature', x.shape)
                x = self.reshape(x)
                # print('reshape', x.shape) 
                module_outputs.append(x)

            # print('module_outputs', len(module_outputs), module_outputs[0].shape) # module_outputs 16 torch.Size([8, 25, 1024])
            
            feat = torch.cat(module_outputs, 1) #(hw*FRR) x c
            
            # Change to covariance matrix
            feat = torch.bmm(feat.transpose(1,2), feat)
            # zero = torch.zeros(feat.shape[0], 400, 624).to(self.device)
            # feat = torch.cat((feat, zero), 2)
            
            # Correlation matrix
            # feat = feat.transpose(1,2)
            # new_feat = torch.corrcoef(feat[0])
            # new_feat = torch.unsqueeze(new_feat, 0)
            # for i in range(1,feat.shape[0]):
                # temp = torch.corrcoef(feat[i])
                # temp = torch.unsqueeze(temp, 0)
                # new_feat = torch.cat((new_feat, temp),0)
            # print(new_feat.shape)
            # feat = new_feat
            # zero = torch.zeros(feat.shape[0], 400, 624).to(self.device)
            # feat = torch.cat((feat, zero), 2)
            
            # Plot                
            # new_feat = new_feat[1].data.cpu().numpy()                    
            # plt.figure(figsize=(30,20))
            # heatmap = sns.heatmap(new_feat[0:512,0:512], vmin=-0, vmax=0.7, cmap="YlOrBr")
            # plt.show()
            
            # print(type(feat))
            
            # sts()
            # print('before svt',feat.shape) # before svt torch.Size([8, 400, 1024])
            if self.heads != 0:
                feat, feat_seq = self.spatial_transformer(feat,None,True)
            elif self.heads == 0:
                feat, feat_seq = self.spatial_transformer(feat)
            # print('feat',feat.shape, 'feat_seq', feat_seq.shape) # feat torch.Size([8, 1024]) feat_seq torch.Size([8, 401, 1024])

            if self.percent !=0:
                mask_frame_num = X['mask_frame_num']
                # print('mask_frame_num', mask_frame_num)
                a_arr = np.array([t.numpy() for t in mask_frame_num])
                # print('a_arr', a_arr)
                mask_num = a_arr.transpose(1,0)
                # print('mask_num', mask_num.shape)

            X = self.inter_feat(feat)
            # print('to cls',X.shape) # to cls torch.Size([8, 4096])
            
            if self.masked_reconstruction:
                rec_seq = []
                feature_seq = []
                feat_seq = feat_seq[:, 1:]
                # print('re feat_seq', feat_seq.shape) #[1, 400, 1024]
                each_frame=feat_seq.shape[1]//self.FRR
                for z in range(self.FRR):
                    ind = z*self.patch_size
                    # mask background part feature to 0 matrix
                    if self.use_mask:
                        if self.percent !=0:
                            f_list = []
                            for num, a in enumerate(mask_num):
                                if z in a:
                                    fea = torch.zeros(25, 1024).to(self.device, non_blocking=True)
                                    # print('mask fea',fea.shape)
                                else:
                                    fea = feat_seq[num, ind:ind+self.patch_size]
                                    # print('unmask fea',fea.shape)
                                f_list.append(fea)
                            f = torch.stack(f_list, dim=0)
                        else:
                            f = feat_seq[:, ind:ind+self.patch_size]
                    else:
                        f = feat_seq[:, ind:ind+self.patch_size]
                    # print('f',f.shape) #[8, 25, 1024]
                    f = self.recover(f)
                    # print('f',f.shape) # [8, 1024, 5, 5]
                    feature_seq.append(f)
                    # print('recover f',f.shape) # (8,1024,5,5)
                    rec_seq.append(self.gen(f))
                # print('gan rec_seq',len(rec_seq), rec_seq[0].shape) # gan rec_seq 16 torch.Size([8, 3, 144, 144])
                rec_seq = torch.stack(rec_seq).permute(1,0,2,3,4)
                # print('rec_seq', rec_seq.shape) # rec_seq torch.Size([8, 16, 3, 144, 144])
            else:
                rec_seq = []
            # Concatenate current image CNN output 
            return X, rec_seq#, feature_seq
            # return X, rec_seq
        else:
            raise
            
        
    def forward(self, X, forward_mode='train'):
        
        X=self.stforward(X)
        # print('forward X', X[0].shape, X[1].shape) # forward X torch.Size([10,4096]) torch.Size([10,16,3,144,144])
        cls=self.cls(X[0])
        # print('cls', cls.shape) # cls torch.Size([10,2])
        if forward_mode=='test':
        #     return cls, X[1]#, X[2]
        # return cls, X[1]#, X[2]
            return cls
        return cls, X[1]
