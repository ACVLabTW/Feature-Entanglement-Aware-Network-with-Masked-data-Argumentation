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

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        ngf = 4
        nc = 3
        nz = 25
        self.args = args
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.nn.functional.interpolate(x, size=(self.args.crop_size, self.args.crop_size), mode='bilinear', align_corners=False)
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
        self.L1_depth=args.L1_depth
        # self.random_mask = args.random_mask
        self.masked_reconstruction = args.masked_reconstruction
        self.gen = Generator(args) if self.masked_reconstruction else None
        
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
        self.recover =  Rearrange('b c (h w)  -> b c h w', h=32, w=32)

        if 'efficient' in self.backbone:
            shapes = model.extract_features(torch.zeros((1,3,self.crop_size, self.crop_size))).shape
            print(f'Feature map {self.backbone} size is', shapes)
        else:
            shapes = model(torch.zeros((1,3,self.crop_size, self.crop_size))).shape
            print(f'Feature map {self.backbone} size is', shapes)
        num_patches = shapes[-1]*shapes[-2]*args.FRR
        
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

                x = self.reshape(x)
                module_outputs.append(x)


            feat = torch.cat(module_outputs, 1) #(hw*FRR) x c
#             sts()
            feat, feat_seq = self.spatial_transformer(feat,None,True)
            X = self.inter_feat(feat)
            
            if self.masked_reconstruction:
                rec_seq = []
                feat_seq = feat_seq[:, 1:]
                for z in range(self.FRR):
                    ind = z*self.patch_size
                    f = feat_seq[:, z:z+self.patch_size]
                    f = self.recover(f)
    #                 print(f.shape)
                    rec_seq.append(self.gen(f))
                rec_seq = torch.stack(rec_seq).permute(1,0,2,3,4)
            else:
                rec_seq = []
            # Concatenate current image CNN output 
            
            return X, rec_seq
        else:
            raise
            
        
    def forward(self, X, forward_mode='train'):
        X=self.stforward(X)
        cls=self.cls(X[0])
        if forward_mode=='test':
            return cls, X
        return cls, X[1]