from __future__ import print_function, division
import os
import pandas as pd
import h5py

import torch 
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np

import pdb 


def load_features(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        coords = hdf5_file['coords'][:] # array, (629, 2)
        feats = hdf5_file['features'][:].squeeze() # tensor, (629, 512)
        # print('hdf5_file:', hdf5_file['features'][:].shape)
        # print('coords:', coords.shape)
        # print('feats:', feats.shape)
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return coords, feats

class SlideDataset(Dataset):
    def __init__(self, dataset_name, csv_path, high_features_path, low_features_path, modalities, embedding_size=None, sample=-1, train=True):
        """
        Args:
            dataset_name (string) : name of dataset for differential handling 
            csv_path (string): Path to the csv file with labels and slide_id.
            high_features_path (string): Directory with all the feature files. (高倍率小图)
            low_features_path (string): Directory with all the feature files. (低倍率大图)
            sample (int): Number of tokens to sample per modality. Default: no sampling. 
            modalities (string): he or all. 
        """
        self.dataset_name = dataset_name
        self.dataframe = pd.read_csv(csv_path)
        self.high_features_path = high_features_path
        self.low_features_path = low_features_path
        self.modalities = modalities
        self.sample = sample
        self.train = train
        self.embedding_size = embedding_size

    def __len__(self):
        return len(self.dataframe)
    
    def sample_n(self, feats): # 随机抽取self.sample个
        if self.sample > -1:
            if len(feats) < self.sample:
                patch_indices = torch.randint(0, len(feats), (self.sample,))
                feats = [feats[i] for i in patch_indices] # TODO:
            else:
                patch_indices = torch.randperm(len(feats))[:self.sample]
                feats = [feats[i] for i in patch_indices]
        return feats

    def __getitem__(self, index):
        # common to all datasets
        slide_id = self.dataframe.iloc[index, self.dataframe.columns.get_loc('slide_id')]
        modality_labels = [self.dataframe.iloc[index, self.dataframe.columns.get_loc(modality)] for modality in self.modalities]
        multi_scale_data = []
        if self.train:
            
            split_type = self.dataframe.iloc[index, self.dataframe.columns.get_loc('split')]
            special_id = "" if split_type == "train" else f"_{split_type}"
            
            all_high_feats = []
            all_low_feats = []
            # print('self.modalities:',self.modalities)  ['HE', 'HER2', 'PGR', 'KI67', 'ER']
            # pdb.set_trace()
            for modality, modality_label in zip(self.modalities, modality_labels):
                curr_high_h5_path = os.path.join(self.high_features_path, f"{slide_id}_{modality}{special_id}.h5")
                curr_low_h5_path = os.path.join(self.low_features_path, f"{slide_id}_{modality}{special_id}.h5")
                curr_high_coords, curr_high_feats = load_features(curr_high_h5_path) if modality_label == 1 else (torch.zeros([2, 2]), torch.zeros([2, self.embedding_size]))
                curr_low_coords, curr_low_feats = load_features(curr_low_h5_path) if modality_label == 1 else (torch.zeros([2, 2]), torch.zeros([2, self.embedding_size]))
                # 对高倍率处理对坐标和特征进行排序
                high_sorted_indices = np.lexsort((curr_high_coords[:, 1], curr_high_coords[:, 0]))  # 先按x升序排，再按y升序排
                curr_high_coords = curr_high_coords[high_sorted_indices]
                curr_high_feats = curr_high_feats[high_sorted_indices]
                # curr_high_feats = self.sample_n(curr_high_feats)
                all_high_feats.append(curr_high_feats)
                # 对低倍率处理对坐标和特征进行排序
                low_sorted_indices = np.lexsort((curr_low_coords[:, 1], curr_low_coords[:, 0]))  # 先按x升序排，再按y升序排
                curr_low_coords = curr_low_coords[low_sorted_indices]
                curr_low_feats = curr_low_feats[low_sorted_indices]
                # curr_low_feats = self.sample_n(curr_low_feats)
                all_low_feats.append(curr_low_feats)

                # 建立对应关系
                '''
                multi_scale_data: 
                一个关于模态的list
                    每个里面是一个list
                        list中的每一个元素是一个关于大patch的dict
                        {'low_feat': 单个的feat，curr_low_feats[i],
                        'low_coords': 单个的feat，curr_low_coords[i],
                        'high_feat': [ # 左上，右上，左下，右下
                            {'feat':curr_low_feats[j], # j从4i到4i+3
                            'coords':curr_low_coords[j] # j从4i到4i+3
                            },
                            ....
                        ] 
                        }
                对应关系是根据坐标来看的。比如一个大patch(low)是由四个小patch(high)组成。
                具体的，比如第一个大patch的坐标是[4672, 11968],对应的4个小patch分别是（其实就是前4个小patch）:
                [[ 4672, 11968],
                [ 4672, 12224],
                [ 4928, 11968],
                [ 4928, 12224]]
                '''
                # 建立对应关系
                multi_scale_data_modal = []
                for i in range(len(curr_low_coords)):
                    low_coord = curr_low_coords[i]
                    low_feat = curr_low_feats[i]
                    # 获取high的coords
                    high_coords_1 = low_coord
                    high_coords_2 = low_coord + np.array([0, 256])
                    high_coords_3 = low_coord + np.array([256, 0])
                    high_coords_4 = low_coord + np.array([256, 256])
                    high_coords_list = [high_coords_1, high_coords_2, high_coords_3, high_coords_4]
                    high_patch = []
                    for coord in high_coords_list:
                        # 查找与 coord 相等的行索引
                        index2 = np.where((curr_high_coords == coord).all(axis=1))[0]
                        high_patch.append(
                            {'feat': curr_high_feats[index2].squeeze(), 'coords': curr_high_coords[index2].squeeze()}
                        )
                        # if len(index2) > 0:  # 确保找到了匹配的坐标
                        #     feat = curr_high_feats[index2[0]]  # 只取第一个匹配的特征
                        #     if len(feat.shape) == 1:
                        #         feat = feat.unsqueeze(0)  # 确保特征维度统一为 [1, 512]
                        #     high_patch.append(
                        #         {'feat': feat.squeeze(), 'coords': curr_high_coords[index2[0]]}
                        #     )
                            # print('high_patch:', curr_high_feats[index2].squeeze().shape)

                    multi_scale_data_modal.append({
                        'low_feat': low_feat,
                        'low_coords': low_coord,
                        'high_feat': high_patch
                    })


                # 对multi_scale_data_modal做resample
                multi_scale_data_modal = self.sample_n(multi_scale_data_modal)
                    
                multi_scale_data.append(multi_scale_data_modal) # 含有多个模态的


                
        else:
            curr_high_h5_path = os.path.join(self.high_features_path, f"{slide_id}.h5")
            curr_low_h5_path = os.path.join(self.low_features_path, f"{slide_id}.h5")
            curr_high_feats = load_features(curr_high_h5_path)
            curr_low_feats = load_features(curr_low_h5_path)
            all_high_feats = [curr_high_feats]
            all_low_feats = [curr_low_feats]
            modality_labels = [1]
            
        data = {
            'multi_scale_datas': multi_scale_data, # 见上注释
            'modality_labels': modality_labels, # 包含同一个slide的多个模态的labels
            'slide_id': slide_id # 同一个slide的id
        }
        
        return data

def collate(batch):
        # Create separate lists for features and labels
        slide_ids = [item['slide_id'] for item in batch]
        multi_scale_datas = [item['multi_scale_datas'] for item in batch]
        batch_labels = [torch.Tensor(item['modality_labels']) for item in batch]
        batch_labels_stacked = torch.stack(batch_labels)

        all_wsi_feats = []
        for bs_idx, multi_scale_data in enumerate(multi_scale_datas):
            # bs是一个维度
            mix_feature_per_batch = []
            for modal_idx, multi_scale_data_per_modal in enumerate(multi_scale_data):
                # modal是一个维度
                mix_feature_per_modal = []
                for patch_idx, multi_scale_data_per_patch in enumerate(multi_scale_data_per_modal):
                    low_feature = multi_scale_data_per_patch['low_feat']
                    high_feature = [high_data['feat'] for high_data in multi_scale_data_per_patch['high_feat']]
                    high_feature = torch.stack(high_feature)
                    # 将low_feature扩展到与high_feature相同的维度
                    low_feature_expanded = low_feature.unsqueeze(0).expand_as(high_feature) 
                    # 拼接特征
                    mix_feature_per_modal.append(torch.cat((high_feature, low_feature_expanded), dim=-1))  # [4, 1024]

                mix_feature_per_modal = torch.cat(mix_feature_per_modal, dim=0)  
                mix_feature_per_batch.append(mix_feature_per_modal)
            mix_feature_per_batch = torch.stack(mix_feature_per_batch) # [5, c
            all_wsi_feats.append(mix_feature_per_batch)
        all_wsi_feats = torch.stack(all_wsi_feats) # [10, 5, 2048, 342]

        return {
            "all_wsi_feats" : all_wsi_feats,
            "modality_labels" : batch_labels_stacked,
            'slide_ids' : slide_ids
        }


class SimpleDataset(Dataset):
    def __init__(self, high_features_dir, low_features_dir):
        """
        Args:
            features_path (string): Directory with all the feature files.
        """
        self.high_features_dir = high_features_dir
        self.low_features_dir = low_features_dir
        self.high_fnames = os.listdir(self.high_features_dir)
        self.high_fnames = [fn for fn in self.high_fnames if fn.endswith('.h5')]
        self.low_fnames = os.listdir(self.low_features_dir)
        self.low_fnames = [fn for fn in self.low_fnames if fn.endswith('.h5')]

    def __len__(self):
        return len(self.high_fnames)
    
    def __getitem__(self, index):
        # print('index:', index)
        curr_high_h5_path =  os.path.join(self.high_features_dir, self.high_fnames[index])
        curr_low_h5_path = os.path.join(self.low_features_dir, self.low_fnames[index])
        curr_high_coords, curr_high_feats = load_features(curr_high_h5_path)
        # print('curr_high_feats:', curr_high_feats.shape)
        curr_low_coords, curr_low_feats = load_features(curr_low_h5_path)
        # print('curr_low_feats:', curr_low_feats.shape)
        # 对高倍率处理对坐标和特征进行排序
        high_sorted_indices = np.lexsort((curr_high_coords[:, 1], curr_high_coords[:, 0]))  # 先按x升序排，再按y升序排
        curr_high_coords = curr_high_coords[high_sorted_indices]
        curr_high_feats = curr_high_feats[high_sorted_indices]
        # 对低倍率处理对坐标和特征进行排序
        low_sorted_indices = np.lexsort((curr_low_coords[:, 1], curr_low_coords[:, 0]))  # 先按x升序排，再按y升序排
        curr_low_coords = curr_low_coords[low_sorted_indices]
        curr_low_feats = curr_low_feats[low_sorted_indices]
        multi_scale_data_modal = []
        for i in range(len(curr_low_coords)):
            low_coord = curr_low_coords[i]
            low_feat = curr_low_feats[i]
            # 获取high的coords
            high_coords_1 = low_coord
            high_coords_2 = low_coord + np.array([0, 256])
            high_coords_3 = low_coord + np.array([256, 0])
            high_coords_4 = low_coord + np.array([256, 256])
            high_coords_list = [high_coords_1, high_coords_2, high_coords_3, high_coords_4]
            high_patch = []
            for coord in high_coords_list:
                # 查找与 coord 相等的行索引
                index2 = np.where((curr_high_coords == coord).all(axis=1))[0]
                high_patch.append(
                    {'feat': curr_high_feats[index2].squeeze(), 'coords': curr_high_coords[index2].squeeze()}
                )
                # if len(index2) > 0:  # 确保找到了匹配的坐标
                #     feat = curr_high_feats[index2[0]]  # 只取第一个匹配的特征
                #     if len(feat.shape) == 1:
                #         feat = feat.unsqueeze(0)  # 确保特征维度统一为 [1, 512]
                #     high_patch.append(
                #         {'feat': feat.squeeze(), 'coords': curr_high_coords[index2[0]]}
                #     )
                    # print('high_patch:', curr_high_feats[index2].squeeze().shape)

            multi_scale_data_modal.append({
                'low_feat': low_feat,
                'low_coords': low_coord,
                'high_feat': high_patch
            })
        slide_id = os.path.splitext(self.high_fnames[index])[0] 
        # print('slide_id:', slide_id)
        return multi_scale_data_modal, slide_id


def simple_collate(batch):
    multi_scale_data_modal, slide_ids = zip(*batch)
    all_wsi_feats = []
    all_low_coords = []
    all_high_coords = []
    
    for bs_idx, multi_scale_data in enumerate(multi_scale_data_modal):
        mix_feature_per_batch = []
        low_coords_batch = []
        high_coords_batch = []
        
        for patch_idx, multi_scale_data_per_patch in enumerate(multi_scale_data):
            low_feature = multi_scale_data_per_patch['low_feat']
            low_coords = multi_scale_data_per_patch['low_coords']
            low_coords_batch.append(low_coords)
            
            high_feature = [high_data['feat'] for high_data in multi_scale_data_per_patch['high_feat']]
            high_feature = torch.stack(high_feature)
            high_coords = [high_data['coords'] for high_data in multi_scale_data_per_patch['high_feat']]
            high_coords_batch.extend(high_coords)
            
            low_feature_expanded = low_feature.unsqueeze(0).expand_as(high_feature)
            mix_feature_per_batch.append(torch.cat((high_feature, low_feature_expanded), dim=-1))
            
        mix_feature_per_batch = torch.cat(mix_feature_per_batch, dim=0)
        all_wsi_feats.append(mix_feature_per_batch)
        all_low_coords.append(np.array(low_coords_batch))
        all_high_coords.append(np.array(high_coords_batch))
        
    all_wsi_feats = torch.stack(all_wsi_feats)
    
    return all_wsi_feats, list(slide_ids), all_low_coords, all_high_coords

class BCNBDataset(Dataset):
    def __init__(self, high_features_dir, low_features_dir):
        """
        Args:
            features_path (string): Directory with all the feature files.
        """
        self.high_features_dir = high_features_dir
        self.low_features_dir = low_features_dir
        self.high_fnames = os.listdir(self.high_features_dir)
        self.high_fnames = [fn for fn in self.high_fnames if fn.endswith('.h5')]
        self.low_fnames = os.listdir(self.low_features_dir)
        self.low_fnames = [fn for fn in self.low_fnames if fn.endswith('.h5')]

    def __len__(self):
        return len(self.high_fnames)
    
    def __getitem__(self, index):
        # print('index:', index)
        curr_high_h5_path =  os.path.join(self.high_features_dir, self.high_fnames[index])
        curr_low_h5_path = os.path.join(self.low_features_dir, self.low_fnames[index])
        curr_high_coords, curr_high_feats = load_features(curr_high_h5_path)
        # print('curr_high_feats:', curr_high_feats.shape)
        curr_low_coords, curr_low_feats = load_features(curr_low_h5_path)
        # print('curr_low_feats:', curr_low_feats.shape)
        # 对高倍率处理对坐标和特征进行排序
        high_sorted_indices = np.lexsort((curr_high_coords[:, 1], curr_high_coords[:, 0]))  # 先按x升序排，再按y升序排
        curr_high_coords = curr_high_coords[high_sorted_indices]
        curr_high_feats = curr_high_feats[high_sorted_indices]
        # 对低倍率处理对坐标和特征进行排序
        low_sorted_indices = np.lexsort((curr_low_coords[:, 1], curr_low_coords[:, 0]))  # 先按x升序排，再按y升序排
        curr_low_coords = curr_low_coords[low_sorted_indices]
        curr_low_feats = curr_low_feats[low_sorted_indices]
        multi_scale_data_modal = []
        for i in range(len(curr_low_coords)):
            low_coord = curr_low_coords[i]
            low_feat = curr_low_feats[i]
            # 获取high的coords
            high_coords_1 = low_coord
            high_coords_2 = low_coord + np.array([0, 256])
            high_coords_3 = low_coord + np.array([256, 0])
            high_coords_4 = low_coord + np.array([256, 256])
            high_coords_list = [high_coords_1, high_coords_2, high_coords_3, high_coords_4]
            high_patch = []
            for coord in high_coords_list:
                # 查找与 coord 相等的行索引
                index2 = np.where((curr_high_coords == coord).all(axis=1))[0]
                high_patch.append(
                    {'feat': curr_high_feats[index2].squeeze(), 'coords': curr_high_coords[index2].squeeze()}
                )
                # if len(index2) > 0:  # 确保找到了匹配的坐标
                #     feat = curr_high_feats[index2[0]]  # 只取第一个匹配的特征
                #     if len(feat.shape) == 1:
                #         feat = feat.unsqueeze(0)  # 确保特征维度统一为 [1, 512]
                #     high_patch.append(
                #         {'feat': feat.squeeze(), 'coords': curr_high_coords[index2[0]]}
                #     )
                    # print('high_patch:', curr_high_feats[index2].squeeze().shape)

            multi_scale_data_modal.append({
                'low_feat': low_feat,
                'low_coords': low_coord,
                'high_feat': high_patch
            })
        slide_id = os.path.splitext(self.high_fnames[index])[0]    
        return multi_scale_data_modal, slide_id


def BCNB_collate(batch):
    multi_scale_data_modal, slide_ids = zip(*batch)
    all_wsi_feats = []
    for bs_idx, multi_scale_data in enumerate(multi_scale_data_modal):
        # print(multi_scale_data_per_patch)
        mix_feature_per_batch = []
        for patch_idx, multi_scale_data_per_patch in enumerate(multi_scale_data):
            low_feature = multi_scale_data_per_patch['low_feat']
            # print(low_feature.shape) # torch.Size([512])
            high_feature = [high_data['feat'] for high_data in multi_scale_data_per_patch['high_feat']]
            # print('high_feature:', high_feature)
            high_feature = torch.stack(high_feature)
            
            # 将low_feature扩展到与high_feature相同的维度
            low_feature_expanded = low_feature.unsqueeze(0).expand_as(high_feature)  # [4, 171] -> [4, 171*2]
            # 拼接特征
            mix_feature_per_batch.append(torch.cat((high_feature, low_feature_expanded), dim=-1))  # [4, 171*2]
        mix_feature_per_batch = torch.cat(mix_feature_per_batch, dim=0) # [4, 1024]
        all_wsi_feats.append(mix_feature_per_batch)
        # mix_feature_per_batch.append(mix_feature_per_modal)
    all_wsi_feats = torch.stack(all_wsi_feats) # [1, 4, 1024]
    # print('all_wsi_feats:', all_wsi_feats.shape)
    return all_wsi_feats, list(slide_ids)