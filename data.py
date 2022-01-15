import os
import torch
import numpy as np
# from tqdm import tqdm 
import json, pickle
import random


class TripletDataset(object):
    def __init__(self, root, subdir, npoints=2500):
        self.npoints = npoints
        self.root = root
        self.subdir = subdir
                       
        trainfile = os.path.join(self.root, 'train_test_split', self.subdir, 'shuffled_train_file_list.json')
        validfile = os.path.join(self.root, 'train_test_split', self.subdir, 'shuffled_val_file_list.json')
        testfile = os.path.join(self.root, 'train_test_split', self.subdir, 'shuffled_test_file_list.json')
        
        self.train, self.trn_target = self.getitem(self.subdir, trainfile, 'train')
        self.valid, self.vld_target = self.getitem(self.subdir, validfile, 'valid')
        self.test, self.tst_target = self.getitem(self.subdir, testfile, 'test')
                
        print('Number of train data: {}'.format(len(self.train)))
        print('Number of validation data: {}'.format(len(self.valid)))
        print('Number of test data: {}'.format(len(self.test)))
        
        if os.path.isdir(os.path.join(self.root, 'train_test_split', self.subdir)) == False:
            mkdr_cmd = 'mkdir -p ' + os.path.join(self.root, 'train_test_split', self.subdir)
            os.system(mkdr_cmd)

        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'train.npy'), self.train)
        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'target_train.npy'), self.trn_target)
        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'valid.npy'), self.valid)
        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'target_valid.npy'), self.vld_target)
        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'test.npy'), self.test)
        np.save(os.path.join(self.root, 'train_test_split', self.subdir, 'target_test.npy'), self.tst_target)

    def getitem(self, subdir, splitfile, mode):        
        filelist = json.load(open(splitfile, 'r'))
        datapnts, target = [], []

        print('    Loading {} data...'.format(mode))
        cnt = 0
        for item in filelist.keys():
            cnt += 1
            # print(cnt)
            if cnt%1000 == 0:
                print('\tData count: {}'.format(cnt))

            fn = os.path.join(self.root, subdir, item)
            lbl = filelist[item]
        
            with open(fn+'/t1.json', 'r') as fp:                
                t1 = np.array(json.load(fp)[0], dtype=np.float32)  #.astype(np.float32) 
            with open(fn+'/t2.json', 'r') as fp:
                t2 = np.array(json.load(fp)[0], dtype=np.float32)  #.astype(np.float32) 
            with open(fn+'/t3.json', 'r') as fp:
                t3 = np.array(json.load(fp)[0], dtype=np.float32)  #.astype(np.float32) 
            
            # print('t1.shape: {}, t2.shape:{}, t3.shape: {}'.format(t1.shape,t2.shape,t3.shape))

            # #resample
            npoints = self.npoints
            choice = np.array(range(npoints))
            if npoints > t1.shape[0]:
                zero = [[0,0] for i in range(npoints-t1.shape[0])]
                t1 = np.append(t1, np.array(zero,dtype=np.float32), axis=0)
            else:
                t1 = t1[choice, :]
            if npoints > t2.shape[0]:
                zero = [[0,0] for i in range(npoints-t2.shape[0])]
                t2 = np.append(t2, np.array(zero,dtype=np.float32), axis=0)
            else:
                t2 = t2[choice, :]
            if npoints > t3.shape[0]:
                zero = [[0,0] for i in range(npoints-t3.shape[0])]
                t3 = np.append(t3, np.array(zero,dtype=np.float32), axis=0)
            else:
                t3 = t3[choice, :]

            # print('t1.shape: {}, t2.shape:{}, t3.shape: {}'.format(t1.shape,t2.shape,t3.shape))
            
            # normalization
            t4 = np.concatenate((t1,t2,t3), axis=0)
            x, y = t4[:,0], t4[:,1]
            x_norm = normalize(x.reshape(-1,1), -1, 1)
            y_norm = normalize(y.reshape(-1,1), -1, 1)
            t4 = np.stack((x_norm, y_norm), axis=2).reshape(-1,2)
            
            t1, t2, t3 = t4[0:npoints], t4[npoints:npoints*2], t4[npoints*2:npoints*3]            
            # t1 = torch.from_numpy(t1)
            # t2 = torch.from_numpy(t2)
            # t3 = torch.from_numpy(t3)
            # label = torch.from_numpy(np.array(lbl, dtype=np.float32))
            label = np.array(lbl, dtype=np.float32)
            # print('t1.shape: {}, t2.shape:{}, t3.shape: {}'.format(t1.shape,t2.shape,t3.shape))

            # print('label.shape:', label.shape, label)
            # exit()
            # data = []
            data = np.stack([t1, t2, t3], axis=0)             
            datapnts.append(data)
            target.append(label)
        datapnts, target = np.array(datapnts), np.array(target)
        
        return datapnts, target



class TripletDataset_2Channels(object):
    def __init__(self, root, npoints=2500):
        self.npoints = npoints
        self.root = root
               
        trainfile = os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json')
        validfile = os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json')
        testfile = os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json')
        
        self.train, self.trn_target = self.getitem('GFP', trainfile, 'train')
        self.valid, self.vld_target = self.getitem('GFP', validfile, 'valid')
        self.test, self.tst_target = self.getitem('GFP', testfile, 'test')
                
        print('Number of train data: {}'.format(len(self.train)))
        print('Number of validation data: {}'.format(len(self.valid)))
        print('Number of test data: {}'.format(len(self.test)))
        
        np.save(os.path.join(self.root, 'train_test_split', 'train_triplet.npy'), self.train)
        np.save(os.path.join(self.root, 'train_test_split', 'target_train_triplet.npy'), self.trn_target)
        np.save(os.path.join(self.root, 'train_test_split', 'valid_triplet.npy'), self.valid)
        np.save(os.path.join(self.root, 'train_test_split', 'target_valid_triplet.npy'), self.vld_target)
        np.save(os.path.join(self.root, 'train_test_split', 'test_triplet.npy'), self.test)
        np.save(os.path.join(self.root, 'train_test_split', 'target_test_triplet.npy'), self.tst_target)

    def getitem(self, subdir, splitfile, mode):        
        filelist = json.load(open(splitfile, 'r'))
        datapnts, target = [], []

        print('    Loading {} data...'.format(mode))
        cnt = 0
        for item in filelist.keys():
            cnt += 1
            # print(cnt)
            if cnt%1000 == 0:
                print('\tData count: {}'.format(cnt))

            fn = os.path.join(self.root, subdir, item)
            lbl = filelist[item]
        
            with open(fn+'/t1.json', 'r') as fp:
                t1 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
            with open(fn+'/t2.json', 'r') as fp:
                t2 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
            with open(fn+'/t3.json', 'r') as fp:
                t3 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
        
            # #resample
            choice = np.array(range(self.npoints))
            t1 = t1[choice, :]
            t2 = t2[choice, :]
            t3 = t3[choice, :]

            # normalization
            t4 = np.concatenate((t1,t2,t3), axis=0)
            x, y = t4[:,0], t4[:,1]
            x_norm = normalize(x.reshape(-1,1), -1, 1)
            y_norm = normalize(y.reshape(-1,1), -1, 1)
            t4 = np.stack((x_norm, y_norm), axis=2).reshape(-1,2)

            npoints = len(choice)
            t1, t2, t3 = t4[0:npoints], t4[npoints:npoints*2], t4[npoints*2:npoints*3]            
            # t1 = torch.from_numpy(t1)
            # t2 = torch.from_numpy(t2)
            # t3 = torch.from_numpy(t3)
            # label = torch.from_numpy(np.array(lbl, dtype=np.float32))
            label = np.array(lbl, dtype=np.float32)
            # print('t1.shape:', t1.shape)
            # print('label.shape:', label.shape, label)
            # exit()
            data = []
            if mode == 'train':                        
                if lbl[0] == 1:
                    # data.append(t2)
                    # data.append(t3)
                    # data.append(t1)
                    data = np.stack([t2, t3, t1], axis=0)                    
                    # data.append(label)
                    # return t2, t3, t1, label
                elif lbl[1] == 1:
                    # data.append(t1)
                    # data.append(t3)
                    # data.append(t2)
                    data = np.stack([t1, t3, t2], axis=0)
                    # data.append(label)
                    # return t1, t3, t2, label
                elif lbl[2] == 1:
                    # data.append(t1)
                    # data.append(t3)
                    # data.append(t2)
                    data = np.stack([t1, t2, t3], axis=0)
                    # data.append(label)
                    # return t1, t2, t3, label
            elif mode == 'val':
                if lbl[0] == 1:
                    # data.append(t2)
                    # data.append(t3)
                    # data.append(t1)
                    data = np.stack([t2, t3, t1], axis=0)
                    # data.append(label)                
                    # return t2, t3, t1, label
                elif lbl[1] == 1:
                    # data.append(t1)
                    # data.append(t3)
                    # data.append(t2)
                    data = np.stack([t1, t3, t2], axis=0)
                    # data.append(label)
                    # return t1, t3, t2, label
                elif lbl[2] == 1:
                    # data.append(t1)
                    # data.append(t2)
                    # data.append(t3)
                    data = np.stack([t1, t3, t2], axis=0)
                    # data.append(label)
                    # return t1, t2, t3, label
            else:
                # print(self.split)
                # data.append(t1)
                # data.append(t2)
                # data.append(t3)
                data = np.stack([t1, t3, t2], axis=0)
                # data.append(label)
                # return t1, t2, t3, label  
            datapnts.append(data)
            target.append(label)
        datapnts, target = np.array(datapnts), np.array(target)
        
        return datapnts, target

    
class TripletDataset_1(object):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.split = split
               
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(self.split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
                
        self.datapath = []
        for item in filelist.keys():
            self.datapath.append((os.path.join(self.root, 'GFP', item), filelist[item]))
        
        print('Number of {} data: {}'.format(self.split, len(self.datapath)))
        # print(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn[0]+'/t1.json', 'r') as fp:
            t1 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
        with open(fn[0]+'/t2.json', 'r') as fp:
            t2 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
        with open(fn[0]+'/t3.json', 'r') as fp:
            t3 = np.array(json.load(fp)[2], dtype=np.float32)  #.astype(np.float32) 
        
        # #resample
        choice = np.array(range(self.npoints))
        t1 = t1[choice, :]
        t2 = t2[choice, :]
        t3 = t3[choice, :]

        t4 = np.concatenate((t1,t2,t3), axis=0)
        x, y = t4[:,0], t4[:,1]

        x_norm = normalize(x.reshape(-1,1), -1, 1)
        y_norm = normalize(y.reshape(-1,1), -1, 1)
        t4 = np.stack((x_norm, y_norm), axis=2).reshape(-1,2)

        npoints = len(choice)
        t1, t2, t3 = t4[0:npoints], t4[npoints:npoints*2], t4[npoints*2:npoints*3]


        t1 = torch.from_numpy(t1)
        t2 = torch.from_numpy(t2)
        t3 = torch.from_numpy(t3)
        label = torch.from_numpy(np.array(fn[1], dtype=np.float32))

        if self.split == 'train':                        
            if fn[1][0] == 1:
                return t2, t3, t1, label
            elif fn[1][1] == 1:
                return t1, t3, t2, label
            elif fn[1][2] == 1:
                return t1, t2, t3, label
        elif self.split == 'val':
            return_choice = random.randint(0,1) 
            if fn[1][0] == 1:                
                return t2, t3, t1, label
            elif fn[1][1] == 1:
                return t1, t3, t2, label
            elif fn[1][2] == 1:
                return t1, t2, t3, label
        else:
            # print(self.split)
            return t1, t2, t3, label        

    def __len__(self):
        return len(self.datapath)



def normalize(X, mn, mx):
    X_std = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    X_scaled = X_std * (mx-mn) + mn
    return X_scaled 
