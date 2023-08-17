import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from simclr.modules import resnet
from vit.vit_pytorch.vit_pytorch import ViT
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()

        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()
    
    
class VViT(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, device, n_features=1024, FRR=8, image_size = 144, SSL='SimCLR', mode='train', bksize=9):
        super(VViT, self).__init__()
        
        self.mode=mode
        self.encoder = resnet.resnet50(pretrained=False)
        
        self.reshape =  Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = bksize, p2 = bksize)
        
        
        featSize = [72,36, 18, 18]
        self.spatial_transformer = ViT(num_patches = (featSize[1]//bksize*4)**2 + (featSize[2]//bksize*16)**2 + (featSize[3]//bksize*16)**2, # Feat map size
                                dim = bksize**2,
                                depth = 3,
                                heads = 8,
                                mlp_dim = 1024,
                                dropout = 0.1,
                                emb_dropout = 0.1)
        
        self.transformer  = ViT(image_size = image_size,
                    dim = 1024,
                    depth = 4,
                    heads = 16,
                    num_patches=FRR,
                    mlp_dim = n_features,
                    dropout = 0.1,
                    emb_dropout = 0.1)
      
        self.device = device
        self.n_features = n_features
        self.FRR=FRR
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if SSL=='SimCLR':
            self.inter_feat = nn.Sequential(
                nn.Linear(1024*FRR+1024, n_features, bias=False),
                nn.GELU(),
                )

            self.projector = nn.Sequential(
                nn.Linear(n_features, n_features)
                )
        elif SSL=='SimSiam':
            
            self.inter_feat = nn.Sequential(
                nn.Linear(1024*FRR+1024, n_features, bias=False),
                nn.GELU(),
                nn.BatchNorm1d(n_features),
                nn.GELU(),
                nn.Linear(n_features, n_features),
                nn.BatchNorm1d(n_features),
                nn.GELU(),
                nn.Linear(n_features, n_features),
                nn.BatchNorm1d(n_features)
                )

            self.projector = nn.Sequential(
                nn.Linear(n_features, n_features),
                nn.BatchNorm1d(n_features),
                nn.GELU(),
                nn.Linear(n_features, n_features),
                )
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
    def features(self, X):
        x2,x3,x4 = self.encoder(X)
#         sts()
        x2 = self.reshape(x2)
        x3 = self.reshape(x3)
        x4 = self.reshape(x4)
        x = torch.cat((x2,x3,x4), 1) # concatenate at second dimension for patch embedding B*num*64
        x = self.spatial_transformer(x) ## B*num*128
        return x
    
    def stforward(self, X):
        lstm_i, module_outputs = [], []
        
        if self.mode=='train':
            for k in range(len(X['img'])):
                x = self.features(X['img'][k].to(self.device, non_blocking=True))
                lstm_i.append(x)
                module_outputs.append(x)

            feat=torch.stack(lstm_i)## time-seq x batch x feat-dim
            feat = feat.permute(1,0,2)
            feat = self.transformer(feat)

            module_outputs.append(feat)
            # Concatenate current image CNN output 
            X = torch.cat(module_outputs, dim=-1)
            X = self.inter_feat(X)
        else:
            tlen = len(X['img'])
            assert tlen>self.FRR
            
            copies = tlen//self.FRR
            Xs = []
            for c in range(copies):
                rem = tlen%self.FRR
                rem = int(np.floor(rem /2))
                ks = c*self.FRR if c<copies else tlen-1-self.FRR
                ke = ks+self.FRR if c < copies else tlen-1
                lstm_i, module_outputs = [], []
                for k in range(ks, ke):
                    x = self.features(X['img'][k].to(self.device, non_blocking=True))

                    lstm_i.append(x)
                    module_outputs.append(x)

                feat=torch.stack(lstm_i)## time-seq x batch x feat-dim
                feat = feat.permute(1,0,2)
                feat = self.transformer(feat)

                module_outputs.append(feat)
                # Concatenate current image CNN output 
                feat = torch.cat(module_outputs, dim=-1)
                feat = self.inter_feat(feat)
                Xs.append(feat)
            X=Xs
            
        return X
        
        
    def forward(self, X, X2):
        if self.mode=='train':
            h1,h2=self.stforward(X), self.stforward(X2)
            z1,z2=self.projector(h1), self.projector(h2)
            return h1,h2,z1,z2    
        else:
            return self.stforward(X)
    
class SingleViT(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, device, n_features=1024, FRR=8, image_size = 144, mode='train', bksize=18, baseFeat=81):
        super(SingleViT, self).__init__()
        
        self.mode=mode
        self.encoder = resnet.resnet50(pretrained=False)
        
        self.reshape =  Rearrange('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = bksize, p2 = bksize)
        
        featSize = [72, 36, 18, 18]
        num_patches = (featSize[2]//bksize*64)**2 + (featSize[3]//bksize*128)**2
        baseFeat_cat = bksize**2
#         print(baseFeat_cat)
        self.spatial_transformer = ViT(num_patches = num_patches, # Feat map size
                                dim = baseFeat_cat,
                                depth = 2,
                                heads = 8,
                                mlp_dim = 1024,
                                dropout = 0.1,
                                emb_dropout = 0.1,
                                pool='mean'
                               )
        
        self.transformer  = ViT(dim = baseFeat_cat,
                    depth = 2,
                    heads = 8,
                    num_patches=FRR,
                    mlp_dim = 1024,
                    dropout = 0.1,
                    emb_dropout = 0.1)
      
        self.device = device
        self.n_features = n_features
        self.FRR=FRR
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
       
        self.inter_feat = nn.Sequential(
            nn.Linear(baseFeat_cat*FRR+baseFeat_cat, n_features, bias=False),
            nn.GELU(),
            )
        
        self.cls = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(n_features//2, n_features//4),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(n_features//4, 2)
            )
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
    
    def features(self, X):
        x3,x4 = self.encoder(X)
#         sts()
#         x2 = self.reshape(x2)
        x3 = self.reshape(x3)
        x4 = self.reshape(x4)
        x = torch.cat((x3,x4), 1) # concatenate at second dimension for patch embedding B*num*64
        x = self.spatial_transformer(x) ## B*num*128
        return x
    
    def stforward(self, X):
        lstm_i, module_outputs = [], []
        
        if self.mode=='train' and len(X['img'])==self.FRR:
            for k in range(len(X['img'])):
                x = self.features(X['img'][k].to(self.device, non_blocking=True))
                
                lstm_i.append(x)
                module_outputs.append(x)

            feat = torch.stack(lstm_i)## time-seq x batch x feat-dim
            feat = feat.permute(1,0,2)
            feat = self.transformer(feat)

            module_outputs.append(feat)
            # Concatenate current image CNN output 
            X = torch.cat(module_outputs, dim=-1)
            X = self.inter_feat(X)
        else:
            tlen = len(X['img'])
            assert tlen>self.FRR
            
            copies = tlen//self.FRR
            Xs = []
            for c in range(copies):
                rem = tlen%self.FRR
                rem = int(np.floor(rem /2))
                ks = c*self.FRR if c<copies else tlen-1-self.FRR
                ke = ks+self.FRR if c < copies else tlen-1
                lstm_i, module_outputs = [], []
                for k in range(ks, ke):
                    x = self.features(X['img'][k].to(self.device, non_blocking=True))
                    
                    lstm_i.append(x)
                    module_outputs.append(x)

                feat=torch.stack(lstm_i)## time-seq x batch x feat-dim
                feat = feat.permute(1,0,2)
                feat = self.transformer(feat)

                module_outputs.append(feat)
                # Concatenate current image CNN output 
                feat = torch.cat(module_outputs, dim=-1)
                feat = self.inter_feat(feat)
                Xs.append(feat)
            X=Xs
            
        return X
        
        
    def forward(self, X, forward_mode='train'):
        X=self.stforward(X)
        if forward_mode=='test':
            for i in range(len(X)):
                X[i] = self.cls(X[i])
        else:
            X=self.cls(X)
        return X