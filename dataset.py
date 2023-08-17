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
from operators import xcsltp
from albumentations.core.transforms_interface import ImageOnlyTransform
import itertools

# for face detect
# from imutils import face_utils
# import imutils
import dlib
import random

IM_SZ=192
CP_SZ=160
random.seed( 10 )
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
    
    def __init__(self, args, ORemove=True, mode='train', filename=None):
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
        # self.random_mask = args.random_mask
        # self.random_mask_pth = glob.glob("/hdd4/FF_src/raw/original_sequences/youtube/raw/*/*.png") if args.random_mask>0 else None
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
        self.percent = args.percent
        self.mydata = args.mydata
        self.log_name = args.log_name

        global CP_SZ
        CP_SZ = args.crop_size
        # self.path_ = os.path.join('No_detect', 'celeb', self.log_name+'.txt')

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
                if self.mydata:
                    certain_set = osj( fn, 'list.txt')
    #                 print(certain_set)
                    if not os.path.isfile(certain_set):
                        print(os.path.isfile(certain_set))
                    fn_list = open(certain_set).readlines()
                    fn_list = [osj(fn, f.strip('\n')) for f in fn_list]
                    [check_fn(f) for f in fn_list]
                else:
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

        self.random_crop = A.Compose([
                                A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.5, 1.0), ratio=(0.9, 1.1), interpolation=1, always_apply=True, p=1.0),
                            ])
  
        
        self.train_transform = A.Compose([
#                                 A.Resize(self.image_size,self.image_size),
                                A.RandomResizedCrop(self.crop_size, self.crop_size, scale=(0.7, 1.1), ratio=(0.9, 1.1), always_apply=True, p=1),
                                A.GaussNoise (var_limit=(0.5/255, 1.0/255), mean=0, per_channel=True, always_apply=False, p=0.5),
                                A.GaussianBlur (blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
                                A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=0.5),
                                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                                A.HorizontalFlip()
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
        print('use MAE')

    def transform(self, img_list, len1, inner_transform=None, fn=None):
        data, label = {}, {}
            
        assert inner_transform is not None
        imgs = []
        t_len = self.FRR + self.marginal
        
        
        x = []
        landmark = []
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

        #MAE
        y1,y2,x1,x2 = 0, 160, 0, 160
        no_detect_num = 0

        for do_mark in x:
            # y1,y2 = 0, do_mark.shape[0]
            # x1,x2 = 0, do_mark.shape[1]

            face_rects = detector((do_mark).astype('uint8'), 0)
            #assert len(face_rects)>0
            #print('face_rects', face_rects[0], 'len', len(face_rects))
            if len(face_rects)==0:
                landmark.append([y1,y2,x1,x2])
                # print('no detect')
                no_detect_num +=1
            else:
                for i, d in enumerate(face_rects):
                    x1 = abs(d.left())
                    y1 = abs(d.top())
                    x2 = abs(d.right())
                    y2 = abs(d.bottom())
                    if y2>do_mark.shape[0]:
                        y2 = do_mark.shape[0]
                    if x2>do_mark.shape[1]:
                        x2 = do_mark.shape[1]
                landmark.append([y1,y2,x1,x2])
            # if y1>=y2 or x1>=x2:
            #     print('y1,y2,x1,x2', y1,y2,x1,x2)
        #print('landmark', len(landmark))

        ## Record the frame number that was not detected
        # if no_detect_num!=0:
        #     with open(self.path_, 'a') as f:
        #         f.write(fn+' '+str(no_detect_num)+'\n')
        #     f.close()

        self.landmark= landmark
        # MAE data augmentation
        #data_aug_type = random.sample((1,2,3),1)[0]

        self.orig_img = x

        # Do_mask
        x = self.face_crop(x, landmark)
        self.x = x.copy()
        if self.percent==0:
            x_mask = x.copy()
        else:
            x_mask, mask_frame_num = self.background_random_crop(x, self.orig_img, landmark, self.image_size, self.percent)
        # mask_ind = np.zeros(len(x))
        # mask_img = []
        # if self.random_mask>0 and self.mode=='train':
        #     for ind, im in enumerate(x):
                
        #         if rn.random() < self.random_mask:
        #             mask_pth = rn.choice(self.random_mask_pth)
        #             im = cv2.imread(mask_pth)
        #             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #             im = self.random_crop(image=im)['image']
        #             mask_img.append(im)
        #             mask_ind[ind] = 1
        
        # x2, x3 = [], []
        # mask_cnt = 0
        # for ind,im in enumerate(x):
        #     im = inner_transform(image=im)['image']
        #     x3.append(im)
            
        #     if self.random_mask>0 and self.mode=='train' and mask_ind[ind]==1:
        #         im = inner_transform(image=mask_img[mask_cnt])['image']
        #         mask_cnt += 1    
        #     x2.append(im)
        # x =np.array(x2)
        # if self.random_mask>0 and self.mode=='train': x2=np.array(x3)
        
        # x = torch.Tensor(x).permute(0, 3, 1, 2)
        # if self.random_mask>0 and self.mode=='train': x2 = torch.Tensor(x2).permute(0, 3, 1, 2)
        x2=[]
        x2_mask=[]
        if inner_transform is not None:
#             data['mosaic'] = torch.Tensor(toImage(x).astype(np.float)).permute(2,0,1)/255.0
            for im, im_mask in zip(self.x, x_mask):
                x2.append(inner_transform(image=im)['image'])
                x2_mask.append(inner_transform(image=im_mask)['image'])
            x=np.array(x2)
            x_mask=np.array(x2_mask)

            
        x = torch.Tensor(x).permute(0, 3, 1, 2)
        x_mask = torch.Tensor(x_mask).permute(0, 3, 1, 2)

        # data['img']={}
        # data['fn'] =fn
        # data['mask'] = mask_ind if self.random_mask>0 and self.mode=='train' else []
        # data['ori_img'] = (x2-127.5)/128.0 if self.random_mask>0 and self.mode=='train' else []
        # for ind, item in enumerate(x):
        #     data['img'][ind] = (item-127.5)/128.0
        data['img']={}
        data['fn'] =fn
        data['mask']={}
        data['ori_img'] = (x-127.5)/128.0 #if self.percent>0 and self.mode!='test' else []

        # use when mask the spatiotemporal features of the background
        # mask_frame = np.repeat(False, self.FRR, axis=None)
        # for i in mask_frame_num:
        #     mask_frame[i]=True
        if self.percent!=0:
            data['mask_frame_num'] = mask_frame_num
        
        for ind, item in enumerate(x):
            data['img'][ind] = (item-127.5)/128.0
          
        for ind, item in enumerate(x_mask):
            data['mask'][ind] = (item-127.5)/128.0
            
        return data
    
    def face_crop(self, x, landmark):
        cut=[]
        # cut face landmark region
        for num, frame_x in enumerate(x):
            top,bottom,left,right = landmark[num]
            cut.append(frame_x[top:bottom,left:right])
        return cut

    def background_random_crop(self, img_allframe, orig_allframe, landmark_allframe, image_size, percent):
        mask_frame_num = random.sample(range(len(img_allframe)), round(percent*len(img_allframe)))
        for num in mask_frame_num:
        # for img, orig, landmark in zip(img_allframe, orig_allframe, landmark_allframe):
            top, bottom, left, right = landmark_allframe[num]
            # print('top, bottom, left, right', top, bottom, left, right)
            x = left
            y = top
            weight = img_allframe[num].shape[1]
            height = img_allframe[num].shape[0]
            half_height = height/4
            half_weight = weight/4
            # print(weight, height, orig_allframe[num].shape[1], orig_allframe[num].shape[0])
            # print(int(left-half_weight),int(right-half_weight), int(top-half_height),int(bottom-half_height))

            ## original edition
            while x in range(left, right) and y in range(top, bottom):
            ## Reduce face coverage edition
            # while x in range(int(left-half_weight),int(right-half_weight)) or y in range(int(top-half_height),int(bottom-half_height)):
                x = random.randint(0, orig_allframe[num].shape[1] - weight)
                y = random.randint(0, orig_allframe[num].shape[0] - height)
            
            img_allframe[num] = orig_allframe[num][y:y+height, x:x+weight]
        return img_allframe, mask_frame_num





    def __getitem__(self, index):
        img_list = self.image_list[index]
        len1 = self.seq_len[index]
        fn = self.fns[index]
        # print(fn)
        transform = self.train_transform if self.mode!='test' else self.test_transform

        im = self.transform(img_list, len1, inner_transform=transform, fn=fn)
        lab = self.label[index]
        

        return im, lab

    def __len__(self):
        return self.n_images
    
