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

# for face detect
from imutils import face_utils
import imutils
import dlib
import random

    
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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
        # face landmark pars
        # self.mask_f_num = args.mask_f_num
        self.percent = args.percent
        self.mydata = args.mydata

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
                self.fn_list = fn_list
                
            if len(fn_list)<self.FRR*self.FREQ+self.marginal:
                print('Pass the filename', fn, 'due to inefficient number of training samples: target:', self.FRR, 'source:', len(fn_list))
                continue

            self.label.append(int(lab.strip('\n')))
            file_list.append(fn_list)
            self.seq_len.append(len(fn_list))
            self.fns.append(fn)

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
#         self.val_transform = A.Compose([
#                                 A.Resize(self.image_size,self.image_size),
#                                 A.RandomCrop(self.crop_size, self.crop_size)
#                                 # A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=0.5),
#                                 # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
#                                 # A.HorizontalFlip()
# #                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
#                             ])
        
#         self.test_transform = A.Compose([
#                                 A.Resize(self.image_size,self.image_size),
#                                 A.CenterCrop(self.crop_size, self.crop_size),
# #                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
#                             ])
        
        self.image_list = file_list

        self.n_images = len(self.image_list)
        if self.local_rank==0:
            print('The # of videos is:', self.n_images, 'on', self.mode, 'mode!')
        print('use MAE')
        
    def transform(self, img_list, len1, inner_transform=None, fn=None):
        data, label = {}, {}
            
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
            # face_rects = detector((im).astype('uint8'), 0)
            # assert len(face_rects)>0
            # #print('face_rects', face_rects)
            # for i, d in enumerate(face_rects):
            #     x1 = abs(d.left())
            #     y1 = abs(d.top())
            #     x2 = abs(d.right())
            #     y2 = abs(d.bottom())
            # landmark.append([y1,y2,x1,x2])

        
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
            # print('x', len(x))

            # landmark = [landmark[index] for index in ind ]
            # landmark = landmark[:self.FRR]
            y1,y2,x1,x2 = 0, 160, 0, 160
            for do_mark in x:
                face_rects = detector((do_mark).astype('uint8'), 0)
                #assert len(face_rects)>0
                #print('face_rects', face_rects[0], 'len', len(face_rects))
                if len(face_rects)==0:
                    landmark.append([y1,y2,x1,x2])
                    print('no detect')
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
                #print('y1,y2,x1,x2', y1,y2,x1,x2)
            #print('landmark', len(landmark))

            
            self.landmark= landmark
            # MAE data augmentation
            #data_aug_type = random.sample((1,2,3),1)[0]
            self.orig_img = x

            # Do_mask
            x = self.face_crop(x, landmark)
            self.x = x

            x_mask = self.background_random_crop(x, self.orig_img, landmark, self.image_size, self.percent)


            #print('data_aug_type', data_aug_type)
            # if data_aug_type==1:
            #     x_mask = self.type1(x, landmark)
            # elif data_aug_type==2:
            #     x_mask = self.type2(x, landmark)
            # elif data_aug_type==3:
            #     x_mask = self.type3(x, landmark)
            # else:
            #     print('do nothing')
            
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
        x2_mask=[]
        if inner_transform is not None:
#             data['mosaic'] = torch.Tensor(toImage(x).astype(np.float)).permute(2,0,1)/255.0
            for im, im_mask in zip(x, x_mask):
                x2.append(inner_transform(image=im)['image'])
                x2_mask.append(inner_transform(image=im_mask)['image'])
            x=np.array(x2)
            x_mask=np.array(x2_mask)

            
        x = torch.Tensor(x).permute(0, 3, 1, 2)
        x_mask = torch.Tensor(x_mask).permute(0, 3, 1, 2)
        data['img']={}
        data['img_mask']={}
        data['img_oig'] = self.orig_img
        data['fn'] =fn
        for ind, item in enumerate(x):
            if self.feature is None:
                #data['img'][ind] = (item-127.5)/128.0
                data['img'][ind] = (item-item.min())/(item.max()-item.min())
            else:
                data['img'][ind] = item/255.0

        for ind, item in enumerate(x_mask):
            if self.feature is None:
                #data['img'][ind] = (item-127.5)/128.0
                data['img_mask'][ind] = (item-item.min())/(item.max()-item.min())
            else:
                data['img_mask'][ind] = item/255.0
  
        return data

    # def get_set_idx(self, patch, landmark_patch, mask_frame_num):
    #     c_p, r_p = patch[mask_frame_num]
    #     y1_p,y2_p,x1_p,x2_p = landmark_patch[mask_frame_num]

    #     face_index_x, face_index_y = np.meshgrid(np.arange(y1_p,(y2_p-1)), np.arange(x1_p,(x2_p-1)))
    #     face_index = np.column_stack([face_index_x.reshape(-1), face_index_y.reshape(-1)])
    #     all_index_x, all_index_y = np.meshgrid(np.arange(c_p-1), np.arange(r_p-1))
    #     all_index = np.column_stack([all_index_x.reshape(-1), all_index_y.reshape(-1)])

    #     all_index_rows = all_index.view([('', all_index.dtype)] * all_index.shape[1])
    #     face_index_rows = face_index.view([('', face_index.dtype)] * face_index.shape[1])
    #     background_index = np.setdiff1d(all_index_rows, face_index_rows).view(all_index.dtype).reshape(-1, all_index.shape[1])
    #     return all_index, face_index, background_index



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
            print(top, bottom, left, right)
            x = left
            y = top
            weight = img_allframe[num].shape[1]
            height = img_allframe[num].shape[0]
            while x in range(left,right) and y in range(top,bottom):
                x = random.randint(0, orig_allframe[num].shape[1] - weight)
                y = random.randint(0, orig_allframe[num].shape[0] - height)
            
            img_allframe[num] = orig_allframe[num][y:y+height, x:x+weight]
        return img_allframe

    def __getitem__(self, index):
        img_list = self.image_list[index]
        len1 = self.seq_len[index]
        fn = self.fns[index]
        
        transform = self.train_transform if self.mode!='test' else self.test_transform

        im = self.transform(img_list, len1, inner_transform=transform, fn=fn)
        lab = self.label[index]

        return im, lab
        # if self.mode=='train':
           
        #     return self.transform(img_list, len1, inner_transform=self.weak_transform, fn=fn), self.transform(img_list, len1, inner_transform=self.strong_transform, fn=fn)
        
        # else:
        #     transform = self.val_transform if self.mode!='test' else self.test_transform
            
        #     im = self.transform(img_list, len1, inner_transform=transform, fn=fn)
        #     lab = self.label[index]
            
        #     return im, lab

    def __len__(self):
        return self.n_images


# class dataset_DFD(torch.utils.data.Dataset):
    
#     def __init__(self, args, ORemove=True, mode='train', filename=None, feature=None):
#         super(dataset_DFD, self).__init__()

#         self.mode        = mode
#         self.root        = args.dataset_dir if mode!='test' and mode!='finetuning' else args.dataset_dir_test
#         self.FRR         = args.FRR
#         self.FREQ        = args.FREQ
#         self.seq_len     = []
#         self.fns         = []
#         self.image_size  = args.image_size
#         self.crop_size   = args.crop_size
#         self.offsets     = (args.image_size-args.crop_size)//2
#         self.random_crop = True
#         self.marginal    = args.marginal if mode!='test' else 0
#         self.label       = []
#         self.local_rank  = 0
#         self.is_denoised = False
#         self.test_aug    = args.test_aug
#         self.ORemove     = ORemove
#         self.FSet        = [int(v) for v in args.MultiFREQ.split(',')] if args.MultiFREQ is not None else None
#         self.maxFREQ     = args.FRR if (self.FSet is None or mode=='test') else max(self.FSet)
#         self.centerCrop  = args.centerCrop/100.0
#         imlist           = args.train_file if mode!='test' else args.test_file
#         self.max_det     = args.max_det
#         self.feature     = feature
#         # face landmark pars
#         self.p = args.p
#         self.mask_f_num = args.mask_f_num
#         self.percent = args.percent

#         if feature is not None:
#             print("Use feature extraction mode")

#         ## Fake one
#         Y=None

#         file_list = []
#         Y=[]
#         if filename is None:
#             filename=imlist
            
#         with open(filename, 'r') as fp:
#             data = fp.readlines()
#         for line in data:
#             fn, lab=line.split(' ')
#             if ORemove:
#                 certain_set = osj(self.root, fn, 'list.txt')
# #                 print(certain_set)
#                 if not os.path.isfile(certain_set):
#                     print(os.path.isfile(certain_set))
#                 fn_list = open(certain_set).readlines()
#                 fn_list = [osj(self.root,fn, f.strip('\n')) for f in fn_list]
#                 [check_fn(f) for f in fn_list]
# #                 print(fn_list)
#             else:
#                 fn_list = glob.glob(osj(self.root, fn, '*.png'))
#                 fn_list = sorting_truncated(fn_list, FRR=self.FRR)
                
#             if len(fn_list)<self.FRR*self.FREQ+self.marginal:
#                 print('Pass the filename', fn, 'due to inefficient number of training samples: target:', self.FRR, 'source:', len(fn_list))
#                 continue

#             self.label.append(int(lab.strip('\n')))
#             file_list.append(fn_list)
#             self.seq_len.append(len(fn_list))
#             self.fns.append(fn)
    
#         self.val_transform = A.Compose([
#                                 A.Resize(self.image_size,self.image_size),
#                                 A.RandomCrop(self.crop_size, self.crop_size)
#                                 # A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=0.5),
#                                 # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
#                                 # A.HorizontalFlip()
# #                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
#                             ])
        
#         self.test_transform = A.Compose([
#                                 A.Resize(self.image_size,self.image_size),
#                                 A.CenterCrop(self.crop_size, self.crop_size),
# #                                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
#                             ])
        
#         self.image_list = file_list

#         self.n_images = len(self.image_list)
#         if self.local_rank==0:
#             print('The # of videos is:', self.n_images, 'on', self.mode, 'mode!')
#         print('use MAE')
        
#     def transform(self, img_list, len1, inner_transform=None, fn=None):
#         data, label = {}, {}
            
#         imgs = []
#         t_len = self.FRR + self.marginal
        
        
#         x = []
#         landmark = []
#         len1 = len(img_list)
#         for ims in img_list:
#             assert os.path.isfile(ims)
#             im = cv2.imread(ims)
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#             x.append(im)
#             # face_rects = detector((im).astype('uint8'), 0)
#             # assert len(face_rects)>0
#             # #print('face_rects', face_rects)
#             # for i, d in enumerate(face_rects):
#             #     x1 = abs(d.left())
#             #     y1 = abs(d.top())
#             #     x2 = abs(d.right())
#             #     y2 = abs(d.bottom())
#             # landmark.append([y1,y2,x1,x2])

        
#         FREQ = self.FREQ if (self.FSet is None or self.mode=='test') else rn.randint(1, self.maxFREQ)
#         if self.mode=='train' or 'val' in self.mode or self.mode=='finetuning':
#             end_point = len1 - self.FRR*FREQ-self.marginal
#             if end_point >= 0:
#                 st_idx = rn.randint(0, end_point)
#                 ind = range(st_idx, st_idx+(FREQ*self.FRR), FREQ)
#             else:
#                 step = int(np.floor(len1/self.FRR))
#                 if step*self.FRR > len1:
#                     ind = range(0, len1)
#                     ind = ind[:self.FRR]
#                 else:
#                     step = 1 if step ==0 else step
#                     ind = range(0, len1, step)
#                     ind = ind[:self.FRR]

#             x = [x[index] for index in ind ]
#             x = x[:self.FRR]
#             assert len(ind) >= self.FRR
#             # print('x', len(x))

#             # landmark = [landmark[index] for index in ind ]
#             # landmark = landmark[:self.FRR]
#             for do_mark in x:
#                 face_rects = detector((do_mark).astype('uint8'), 0)
#                 #assert len(face_rects)>0
#                 #print('face_rects', face_rects[0], 'len', len(face_rects))
#                 if len(face_rects)==0:
#                     landmark.append([y1,y2,x1,x2])
#                     print('no detect')
#                 else:
#                     for i, d in enumerate(face_rects):
#                         x1 = abs(d.left())
#                         y1 = abs(d.top())
#                         x2 = abs(d.right())
#                         y2 = abs(d.bottom())
#                         if y2>do_mark.shape[0]:
#                             y2 = do_mark.shape[0]
#                         if x2>do_mark.shape[1]:
#                             x2 = do_mark.shape[1]
#                     landmark.append([y1,y2,x1,x2])
#                 #print('y1,y2,x1,x2', y1,y2,x1,x2)
#             #print('landmark', len(landmark))

#             # self.x= x
#             # self.landmark= landmark
#             # MAE data augmentation
#             #data_aug_type = random.sample((1,2,3),1)[0]
#             self.orig_img = x

#             # Do_mask
#             x = self.face_crop(x, landmark)
#             x_mask = self.random_mask(x, landmark)

#             #print('data_aug_type', data_aug_type)
#             # if data_aug_type==1:
#             #     x_mask = self.type1(x, landmark)
#             # elif data_aug_type==2:
#             #     x_mask = self.type2(x, landmark)
#             # elif data_aug_type==3:
#             #     x_mask = self.type3(x, landmark)
#             # else:
#             #     print('do nothing')
            
#         elif self.test_aug==0:
            
#             nStep = int(np.floor((len1-1) / self.FRR))
#             max_ind = nStep * self.FRR
#             max_ind = len1 if max_ind>=len1-1 else max_ind
#             ind = list(range(0, max_ind+1, nStep))
#             if len(ind)<self.FRR:
#                 ind = range(self.FRR)
#             x = [x[index] for index in ind ]
#             x = x[:self.FRR]
            
#         elif self.test_aug==1:  # K-crops data augmentation in testing phase
#             max_det=self.max_det
#             if max_det < 99:
            
#                 assert len1>self.FRR*self.FREQ
#                 len11=len1
#                 targetLen = self.FRR*self.FREQ + max_det*self.FREQ
#                 stride = 1 
#                 if len1 > targetLen:
#                     rem = len1 - targetLen
#                     rem = int(np.floor(rem/2))
#                     x = [x[ind] for ind in range(rem,rem+targetLen)]
#                 else:
#                     x = [x[ind] for ind in range(len1)]

#                 len1 = len(x)
#                 if len1<self.FRR:
#                     print('stride',stride)
#                     print('original length',len11)
#                     print('cropped length',len1)
#                     print('remaining length', remaining)

        
#         # Random crop
# #         x2=[]
# #         x2_mask=[]
# #         if inner_transform is not None:
# # #             data['mosaic'] = torch.Tensor(toImage(x).astype(np.float)).permute(2,0,1)/255.0
# #             for im, im_mask in zip(x, x_mask):
# #                 x2.append(inner_transform(image=im)['image'])
# #                 x2_mask.append(inner_transform(image=im_mask)['image'])
# #             x=np.array(x2)
# #             x_mask=np.array(x2_mask)

            
#         x = torch.Tensor(x).permute(0, 3, 1, 2)
#         x_mask = torch.Tensor(x_mask).permute(0, 3, 1, 2)
#         data['img']={}
#         data['img_mask']={}
#         data['img_oig'] = orig_img
#         data['fn'] =fn
#         for ind, item in enumerate(x):
#             if self.feature is None:
#                 #data['img'][ind] = (item-127.5)/128.0
#                 data['img'][ind] = (item-item.min())/(item.max()-item.min())
#             else:
#                 data['img'][ind] = item/255.0

#         for ind, item in enumerate(x_mask):
#             if self.feature is None:
#                 #data['img'][ind] = (item-127.5)/128.0
#                 data['img_mask'][ind] = (item-item.min())/(item.max()-item.min())
#             else:
#                 data['img_mask'][ind] = item/255.0
  
#         return data

#     def get_set_idx(self, patch, landmark_patch, mask_frame_num):
#         c_p, r_p = patch[mask_frame_num]
#         y1_p,y2_p,x1_p,x2_p = landmark_patch[mask_frame_num]

#         face_index_x, face_index_y = np.meshgrid(np.arange(y1_p,(y2_p-1)), np.arange(x1_p,(x2_p-1)))
#         face_index = np.column_stack([face_index_x.reshape(-1), face_index_y.reshape(-1)])
#         all_index_x, all_index_y = np.meshgrid(np.arange(c_p-1), np.arange(r_p-1))
#         all_index = np.column_stack([all_index_x.reshape(-1), all_index_y.reshape(-1)])

#         all_index_rows = all_index.view([('', all_index.dtype)] * all_index.shape[1])
#         face_index_rows = face_index.view([('', face_index.dtype)] * face_index.shape[1])
#         background_index = np.setdiff1d(all_index_rows, face_index_rows).view(all_index.dtype).reshape(-1, all_index.shape[1])
#         return all_index, face_index, background_index



#     def face_crop(self, x, landmark):
#         cut=[]
#         # cut face landmark region
#         for num, frame_x in enumerate(x):
#             top,bottom,left,right = landmark[num]
#             cut.append(frame_x[top:bottom,left:right])
#         return cut

#     def random_mask(self, x, landmark):
#         # take one of 16 frame become noise(select from region out of face landmark)
#         patch = list(map(lambda x: [x.shape[0]//self.p, x.shape[1]//self.p], x))
#         landmark_patch = list(map(lambda x: [x[0]//self.p, x[1]//self.p, x[2]//self.p, x[3]//self.p], landmark))

#         cut=[]
#         for num, frame_x in enumerate(x):
#             top,bottom,left,right = landmark_patch[num]
#             cut.append(frame_x[top*self.p:(bottom)*self.p, left*self.p:(right)*self.p])

#         mask_frame_num = random.sample(range(len(x)),1)[0]

#         all_index, face_index, background_index = self.get_set_idx(patch, landmark_patch, mask_frame_num)

#         if len(background_index)<=30:
#             pass
#         else:
#             mask_background_p = random.choices(background_index.tolist(),k=len(face_index))
#             #x_ = x
#             for num, patch_idx in enumerate(face_index):
#                 c,r = patch_idx
#                 c_m,r_m = mask_background_p[num]
#                 x[mask_frame_num][c*self.p:(c+1)*self.p, r*self.p:(r+1)*self.p] = x[mask_frame_num][c_m*self.p:(c_m+1)*self.p, r_m*self.p:(r_m+1)*self.p]
#             top,bottom,left,right = landmark_patch[mask_frame_num]
#             cut[mask_frame_num] = x[mask_frame_num][top*self.p:(bottom)*self.p,left*self.p:(right)*self.p]
#         return cut
#     # data augmentation type
#     # def type1(self, x, landmark):
#     #     cut=[]
#     #     # cut face landmark region
#     #     for num, frame_x in enumerate(x):
#     #         top,bottom,left,right = landmark[num]
#     #         cut.append(frame_x[top:bottom,left:right])
#     #     return cut

#     # def type2(self, x, landmark):
#     #     # take one of 16 frame become noise(select from region out of face landmark)
#     #     patch = list(map(lambda x: [x.shape[0]//self.p, x.shape[1]//self.p], x))
#     #     landmark_patch = list(map(lambda x: [x[0]//self.p, x[1]//self.p, x[2]//self.p, x[3]//self.p], landmark))

#     #     cut=[]
#     #     for num, frame_x in enumerate(x):
#     #         top,bottom,left,right = landmark_patch[num]
#     #         cut.append(frame_x[top*self.p:(bottom)*self.p, left*self.p:(right)*self.p])

#     #     mask_frame_num = random.sample(range(len(x)),1)[0]

#     #     all_index, face_index, background_index = self.get_set_idx(patch, landmark_patch, mask_frame_num)

#     #     if len(background_index)<=30:
#     #         pass
#     #     else:
#     #         mask_background_p = random.choices(background_index.tolist(),k=len(face_index))
#     #         #x_ = x
#     #         for num, patch_idx in enumerate(face_index):
#     #             c,r = patch_idx
#     #             c_m,r_m = mask_background_p[num]
#     #             x[mask_frame_num][c*self.p:(c+1)*self.p, r*self.p:(r+1)*self.p] = x[mask_frame_num][c_m*self.p:(c_m+1)*self.p, r_m*self.p:(r_m+1)*self.p]
#     #         top,bottom,left,right = landmark_patch[mask_frame_num]
#     #         cut[mask_frame_num] = x[mask_frame_num][top*self.p:(bottom)*self.p,left*self.p:(right)*self.p]
#     #     return cut
    
#     # def type3(self, x, landmark):
#     #     # take few frame(self.mask_f_num) of 16 frame become part of noise(self.percent)(select from region out of face landmark)
#     #     patch = list(map(lambda x: [x.shape[0]//self.p, x.shape[1]//self.p], x))
#     #     landmark_patch = list(map(lambda x: [x[0]//self.p, x[1]//self.p, x[2]//self.p, x[3]//self.p], landmark))

#     #     cut=[]
#     #     for num, frame_x in enumerate(x):
#     #         top,bottom,left,right = landmark_patch[num]
#     #         cut.append(frame_x[top*self.p:(bottom)*self.p, left*self.p:(right)*self.p])
        
#     #     mask_frame_nums = random.sample(range(len(x)), self.mask_f_num)

#     #     for mask_frame_num in mask_frame_nums:
            
#     #         all_index, face_index, background_index = self.get_set_idx(patch, landmark_patch, mask_frame_num)
#     #         if len(background_index)<=30:
#     #             pass
#     #         else:
#     #             mask_background_p = random.choices(background_index.tolist(),k=round(len(face_index)*self.percent))
#     #             mask_face_p = random.sample(face_index.tolist(),round(len(face_index)*self.percent))
#     #             #x_ = x
#     #             for num, patch_idx in enumerate(mask_face_p):
#     #                 c,r = patch_idx
#     #                 c_m,r_m = mask_background_p[num]
#     #                 if x[mask_frame_num][c*self.p:(c+1)*self.p, r*self.p:(r+1)*self.p].shape!=(16,16,3):
#     #                     print(x[mask_frame_num].shape, c,r, c_m,r_m, all_index, face_index, background_index, patch[mask_frame_num], landmark_patch[mask_frame_num], landmark[mask_frame_num])
#     #                 x[mask_frame_num][c*self.p:(c+1)*self.p, r*self.p:(r+1)*self.p] = x[mask_frame_num][c_m*self.p:(c_m+1)*self.p, r_m*self.p:(r_m+1)*self.p]

#     #             top,bottom,left,right = landmark_patch[mask_frame_num]
#     #             cut[mask_frame_num] = x[mask_frame_num][top*self.p:(bottom)*self.p, left*self.p:(right)*self.p]
#     #     return cut

#     def __getitem__(self, index):
#         img_list = self.image_list[index]
#         len1 = self.seq_len[index]
#         fn = self.fns[index]
            
#         if self.mode=='train':
           
#             return self.transform(img_list, len1, inner_transform=self.weak_transform, fn=fn), self.transform(img_list, len1, inner_transform=self.strong_transform, fn=fn)
        
#         else:
#             transform = self.val_transform if self.mode!='test' else self.test_transform
#             im = self.transform(img_list, len1, inner_transform=transform, fn=fn)
#             lab = self.label[index]
            
#             return im, lab

#     def __len__(self):
#         return self.n_images