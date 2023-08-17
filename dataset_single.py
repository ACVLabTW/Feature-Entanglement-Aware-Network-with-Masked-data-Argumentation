import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
import cv2
import os, glob, numpy as np
from os.path import join as osj
from skimage.restoration import denoise_wavelet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.utils import save_image
from skimage.feature import local_binary_pattern, hog


def lbp_transform(x, **kwargs):
    radius = 2
    n_points = 8 * radius
    METHOD = 'ror'
    x = np.array(x).astype(np.float32)
    for b in range(x.shape[0]):
        for i in range(x.shape[3]):
            x[b,:,:,i] = local_binary_pattern(x[b,:, :,i], n_points, radius, METHOD)
    return x

    
def sorting_truncated(fn, FRR=8):
    len1=len(fn)
    st, et = 0, 300
        
    fns = []
    basename = '/'.join(fn[0].split('/')[:-1])

    for i in range(st, et+1):
        path=osj(basename, str(i)+'.png')
        if os.path.isfile(path):
            fns.append(path)
#         else:
#             print('Something wrong!!', path)
    return fns
def check_fn(filename):
    assert os.path.isfile(filename)

def toList(ims, size):

    h,w = size
    h1,w1,_ = ims.shape
    FRR = h1//h * w1//w
    images = []
    for idx in range(FRR):
        widx = idx%2* w
        hidx = idx//2* h
        images.append(ims[hidx:hidx+h, widx:widx+w, :])
    images = np.array(images)
    return images
    
def toImage(images):
    if len(images)%2 != 0:
        print('Failed to process!!')
    else:
        h,w,c = images[0].shape
        wlen = len(images)//2
        ims = np.zeros((h*wlen, w*2, c), np.uint8)
        for idx, im in enumerate(images):
            widx = idx%2 * w
            hidx = idx//2 * h
            ims[hidx:hidx+h, widx:widx+w, :] = im
        return ims

class dataset_DFD(torch.utils.data.Dataset):
    
    def __init__(self, args, ORemove=True, mode='train', filename=None, feature=None):
        super(dataset_DFD, self).__init__()

        self.mode        = mode
        self.root        = args.dataset_dir if mode!='test' and mode!='finetuning' else args.dataset_dir_test
        self.FRR         = args.FRR
        self.FREQ        = args.FREQ
        self.seq_len     = []
        self.fns         = []
        self.image_size  = args.image_size
        self.crop_size   = args.crop_size
        self.offsets     = (args.image_size-args.crop_size)//2
        self.random_crop = True
        self.marginal    = args.marginal if mode!='test' else 0
        self.label       = []
        self.local_rank  = 0
        self.is_denoised = False
        self.test_aug    = args.test_aug
        self.ORemove     = ORemove
        self.FSet        = [int(v) for v in args.MultiFREQ.split(',')] if args.MultiFREQ is not None else None
        self.maxFREQ     = args.FRR if (self.FSet is None or mode=='test') else max(self.FSet)
        self.centerCrop  = args.centerCrop/100.0
        imlist           = args.train_file if mode!='test' else args.test_file
        self.max_det     = args.max_det
        self.feature     = feature
        if feature is not None:
            print("Use feature extraction mode")

        ## Fake one
        Y=None

        file_list = []
        Y=[]
        if filename is None:
            filename=imlist
            
        with open(filename, 'r') as fp:
            data = fp.readlines()
        for line in data:
            fn, lab=line.split(' ')
            if ORemove:
                certain_set = osj(self.root, fn, 'list.txt')
#                 print(certain_set)
                if not os.path.isfile(certain_set):
                    print(os.path.isfile(certain_set))
                fn_list = open(certain_set).readlines()
                fn_list = [osj(self.root,fn, f.strip('\n')) for f in fn_list]
                [check_fn(f) for f in fn_list]
#                 print(fn_list)
            else:
                fn_list = glob.glob(osj(self.root, fn, '*.png'))
                fn_list = sorting_truncated(fn_list, FRR=self.FRR)
                
            if len(fn_list)<self.FRR*self.FREQ+self.marginal:
                print('Pass the filename', fn, 'due to inefficient number of training samples: target:', self.FRR, 'source:', len(fn_list))
                continue

            self.label.append(int(lab.strip('\n')))
            file_list.append(fn_list)
            self.seq_len.append(len(fn_list))
            self.fns.append(fn)


        self.weak_transform = A.Compose([
                                A.Resize(self.image_size,self.image_size),
                                A.RandomCrop(self.crop_size, self.crop_size),
                                A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=0.5),
                                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                                A.HorizontalFlip()
#                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
                            ])
        
        self.strong_transform = A.Compose([
                                A.Resize(self.image_size,self.image_size),
                                A.RandomCrop(self.crop_size, self.crop_size),
                                A.RandomBrightnessContrast(brightness_limit = 0.4, contrast_limit = 0.4, p=0.8),
                                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.8),
                                A.HorizontalFlip()
#                                 A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
                            ])
        
        self.val_transform = A.Compose([
                                A.Resize(self.image_size,self.image_size),
                                A.RandomCrop(self.crop_size, self.crop_size),
                                A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=0.5),
                                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                                A.HorizontalFlip()
#                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
                            ])
        
        self.test_transform = A.Compose([
                                A.Resize(self.image_size,self.image_size),
                                A.CenterCrop(self.crop_size, self.crop_size),
#                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
                            ])
        
        self.image_list = file_list

        self.n_images = len(self.image_list)
        if self.local_rank==0:
            print('The # of videos is:', self.n_images, 'on', self.mode, 'mode!')
        
    def transform(self, img_list, len1, inner_transform=None, fn=None):
        data, label = {}, {}
            
        
        imgs = []
        t_len = self.FRR + self.marginal
        
        
        x = []
        len1 = len(img_list)
        for ims in img_list:
            assert os.path.isfile(ims)
            im = cv2.imread(ims)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            x.append(im)
#             if self.is_denoised:
#                 denoised = denoise_wavelet(im, multichannel=True, convert2ycbcr=True, 
#                                                 method='BayesShrink', mode='soft', rescale_sigma=False)
#                 denoised = im.astype(np.float) - denoised.astype(np.float)
#                 x.append(denoised)

        
#         if self.is_denoised:
#             xd = np.mean(np.concatenate(x, 0), 0)
#             for ind in range(len(x)):
#                 x[ind] = (x[ind] - xd)/50.0
        
        
        FREQ = self.FREQ if (self.FSet is None or self.mode=='test') else rn.randint(1, self.maxFREQ)
        if self.mode=='train' or 'val' in self.mode or self.mode=='finetuning':
            end_point = len1 - self.FRR*FREQ-self.marginal
            if end_point >= 0:
                st_idx = rn.randint(0, end_point)
                ind = range(st_idx, st_idx+(FREQ*self.FRR), FREQ)
            else:
                step = int(np.floor(len1/self.FRR))
                if step*self.FRR > len1:
                    ind = range(0, len1)
                    ind = ind[:self.FRR]
                else:
                    step = 1 if step ==0 else step
                    ind = range(0, len1, step)
                    ind = ind[:self.FRR]

            x = [x[index] for index in ind ]
            x = x[:self.FRR]
            assert len(ind) >= self.FRR

        elif self.test_aug==0:
            
            nStep = int(np.floor((len1-1) / self.FRR))
            max_ind = nStep * self.FRR
            max_ind = len1 if max_ind>=len1-1 else max_ind
            ind = list(range(0, max_ind+1, nStep))
            if len(ind)<self.FRR:
                ind = range(self.FRR)
            x = [x[index] for index in ind ]
            x = x[:self.FRR]
            
        elif self.test_aug==1:  # K-crops data augmentation in testing phase
            max_det=self.max_det
            if max_det < 99:
            
                assert len1>self.FRR*self.FREQ
                len11=len1
                targetLen = self.FRR*self.FREQ + max_det*self.FREQ
                stride = 1 
                if len1 > targetLen:
                    rem = len1 - targetLen
                    rem = int(np.floor(rem/2))
                    x = [x[ind] for ind in range(rem,rem+targetLen)]
                else:
                    x = [x[ind] for ind in range(len1)]

                len1 = len(x)
                if len1<self.FRR:
                    print('stride',stride)
                    print('original length',len11)
                    print('cropped length',len1)
                    print('remaining length', remaining)

        
        # Random crop
        x2=[]
        if inner_transform is not None:
#             data['mosaic'] = torch.Tensor(toImage(x).astype(np.float)).permute(2,0,1)/255.0
            for im in x:
                x2.append(inner_transform(image=im)['image'])
            x=np.array(x2)
#             data['mosaic_transform'] = torch.Tensor(toImage(x).astype(np.float)).permute(2,0,1)/255.0
# #             if not os.path.isfile('img0.png') and self.local_rank==0:
# #                 cv2.imwrite('img0.png', x[3][:,:,::-1].astype(np.uint8))
#             x = toImage(x)
#             data['mosaic'] = torch.Tensor(x.astype(np.float)).permute(2,0,1)/255.0
# #             if not os.path.isfile('img1.png') and self.local_rank==0:
# #                 cv2.imwrite('img1.png', x[:,:,::-1].astype(np.uint8))
#             x = inner_transform(image=x)['image']
#             data['mosaic_transform'] = torch.Tensor(x.astype(np.float)).permute(2,0,1)/255.0
# #             if not os.path.isfile('img2.png') and self.local_rank==0:
# #                 cv2.imwrite('img2.png', x[:,:,::-1].astype(np.uint8))
#             x = toList(x, (self.image_size, self.image_size))
# #             if not os.path.isfile('img3.png') and self.local_rank==0:
# #                 cv2.imwrite('img3.png', x[3][:,:,::-1].astype(np.uint8))
#             x = np.asarray(x, np.float32)
            
#             if self.random_crop:
#                 x2=[]
#                 for z in range( x.shape[0]):
#                     offx, offy = rn.randint(0, self.offsets), rn.randint(0, self.offsets)
#                     w, h = offx+self.crop_size, offy+self.crop_size
#                     x2.append( x[z, offy:h, offx:w, :])
#                 x=np.array(x2)
                
        if self.feature is not None:
            x = lbp_transform(x)
            
        x = torch.Tensor(x).permute(0, 3, 1, 2)
        data['img']={}
        data['fn'] =fn
        for ind, item in enumerate(x):
            if self.feature is None:
                data['img'][ind] = (item-127.5)/128.0
            else:
                data['img'][ind] = item/255.0
            
        return data
        
    def __getitem__(self, index):
        img_list = self.image_list[index]
        len1 = self.seq_len[index]
        fn = self.fns[index]
            
        if self.mode=='train':
           
            return self.transform(img_list, len1, inner_transform=self.weak_transform, fn=fn), self.transform(img_list, len1, inner_transform=self.strong_transform, fn=fn)
        
        else:
            transform = self.val_transform if self.mode!='test' else self.test_transform
            im = self.transform(img_list, len1, inner_transform=transform, fn=fn)
            lab = self.label[index]
            
            return im, lab

    def __len__(self):
        return self.n_images
    
