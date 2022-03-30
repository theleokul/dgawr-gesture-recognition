import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from random import randint,shuffle
import cv2

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.use_data_aug = use_data_aug
        self.data = data

        self.time_len = time_len
        self.compoent_num = 22
        self.depth_max = 50  # depth normalizer
        self.crop_size = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
                
        data_ele = self.data[ind]

        #hand skeleton
        skeleton = np.array(data_ele["skeleton"])
        skeleton_proj = np.array(data_ele["skeleton_proj"])
        gen_info = np.array(data_ele["gen_info"])

        # if self.use_data_aug:
        #     skeleton = self.data_aug(skeleton, depth)

        # sample time_len frames from whole video
        data_num = skeleton.shape[0]
        idx_list = self.sample_frame(data_num)
        
        skeleton = [skeleton[idx] for idx in idx_list]
        skeleton = np.array(skeleton)
        
        skeleton_proj = [skeleton_proj[idx] for idx in idx_list]
        skeleton_proj = np.array(skeleton_proj)
        
        gen_info = [gen_info[idx] for idx in idx_list]
        gen_info = np.array(gen_info)
        
        depth = [cv2.imread(depth[idx], cv2.IMREAD_GRAYSCALE) / self.depth_max for idx in idx_list]
        depth = (np.array(depth).clip(0, 1) - .5) * 2  # [-1, 1]

        # Extend uv to uvd
        skeleton_proj_d = np.empty((self.time_len, self.compoent_num, 1))
        for t in range(self.time_len):
            for k in range(self.compoent_num):
                u, v = skeleton_proj[t, k].astype(int)
                skeleton_proj_d[t, k] = depth[t, v, u]
        skeleton_proj = np.concatenate([skeleton_proj, skeleton_proj_d], axis=-1)
        
        # Crop depth for detection (common crop for all the gesture frames)
        x0_comm, x1_comm = depth.shape[-1], 0
        y0_comm, y1_comm = depth.shape[-2], 0
        for ts in range(self.time_len):
            x, y, width, height = gen_info[ts].astype(int)
            x0, x1 = x, x + width
            y0, y1 = y, y + height
            
            if x0 < x0_comm:
                x0_comm = x0
            if x1 > x1_comm:
                x1_comm = x1
                
            if y0 < y0_comm:
                y0_comm = y0
            if y1 > y1_comm:
                y1_comm = y1
                
        mult_h = (y1_comm - y0_comm) / self.crop_size
        mult_w = (x1_comm - x0_comm)  / self.crop_size
            
        depth_crop = depth[:, y0 : y1, x0 : x1]
        depth_crop = [cv2.resize(depth_crop[ts], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR) for ts in range(self.time_len)]
        depth_crop = np.array(depth_crop)
        
        # Tune skeleton_proj to crop and normalize to [-1, 1]
        skeleton_proj[:, :, 0] = ((skeleton_proj[:, :, 0] - x0) / (x1_comm - x0_comm) - .5) * 2
        skeleton_proj[:, :, 1] = ((skeleton_proj[:, :, 1] - y0) / (y1_comm - y0_comm) - .5) * 2
        skeleton_proj = skeleton_proj.clip(-1, 1)

        skeleton = torch.from_numpy(skeleton).float()
        skeleton_proj = torch.from_numpy(skeleton_proj).float()
        gen_info = torch.from_numpy(gen_info).long()
        depth_crop = torch.from_numpy(depth_crop).float().unsqueeze(1)  # T, [1], H, W
        
        # label
        label = data_ele["label"] - 1

        sample = {
            # Crop
            "skeleton_proj": skeleton_proj,
            "depth": depth_crop,
            "mult_h": mult_h,
            "mult_w": mult_w,
            
            # Raw
            "skeleton": skeleton,
            "gen_info": gen_info,
            "label" : label,
        }

        return sample

    def data_aug(self, skeleton):

        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset
            skeleton = np.array(skeleton)
            return skeleton

        def noise(skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]

            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                    skeleton[t][j_id] += noise_offset
            skeleton = np.array(skeleton)
            return skeleton

        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]#d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i -1] + displace)# r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result

        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)
        if ag_id == 0:
            skeleton = scale(skeleton)
        elif ag_id == 1:
            skeleton = shift(skeleton)
        elif ag_id == 2:
            skeleton = noise(skeleton)
        elif ag_id == 3:
            skeleton = time_interpolate(skeleton)

        return skeleton

    def sample_frame(self, data_num):
        #sample #time_len frames from whole video
        sample_size = self.time_len
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        for i in range(sample_size):
            index = round(each_num * i)
            if index not in idx_list and index < data_num:
                idx_list.append(index)
        idx_list.sort()

        while len(idx_list) < sample_size:
            idx = random.randint(0, data_num - 1)
            if idx not in idx_list:
                idx_list.append(idx)
        idx_list.sort()

        return idx_list
