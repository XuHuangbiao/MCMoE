from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd


class FS1000Dataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = video_feat_path
        self.audio_path = audio_feat_path
        self.flow_path = flow_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        score_idx = {'TES': 130, 'PCS': 60, 'SS': 10, 'TR': 10, 'PE': 10, 'CO': 10, 'IN': 10}
        self.score_range = score_idx[action_type]
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2, 'SS': 3, 'TR': 4, 'PE': 5, 'CO': 6, 'IN': 7}
        labels = []
        score = []
        for i, line in enumerate(fr):
            line = line.strip().split()
            s = float(line[idx[action_type]])
            if action_type == "PCS":
                s = s / float(line[8])
            labels.append([line[0], s])
            score.append(s)
        print("max:", max(score))
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))
        audio_feat = np.load(os.path.join(self.audio_path, self.labels[idx][0] + '.npy'))
        flow_feat = np.load(os.path.join(self.flow_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        video_feat = video_feat.mean(1)
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1])
        # return video_feat, audio_feat, self.labels[idx][1]

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range


class RGDataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = os.path.join(video_feat_path, action_type + '_rgb_VST.npy')
        self.audio_path = os.path.join(audio_feat_path, action_type + '_audio_AST.npy')
        self.flow_path = os.path.join(flow_feat_path, action_type + '_flow_I3D.npy')
        self.erase_path = video_feat_path + '_erTrue'
        self.score_range = 25.
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx['Total_Score']])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(self.video_path, allow_pickle=True).item()[self.labels[idx][0]]
        audio_feat = np.load(self.audio_path, allow_pickle=True).item()[self.labels[idx][0]]
        flow_feat = np.load(self.flow_path, allow_pickle=True).item()[self.labels[idx][0]]
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range
    
    
class FisVDataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='TES', train=True, args=None):
        self.train = train
        self.video_path = os.path.join(video_feat_path, 'FISV_rgb_VST.npy')
        self.audio_path = os.path.join(audio_feat_path, 'FISV_audio_AST.npy')
        self.flow_path = os.path.join(flow_feat_path, 'FISV_flow_I3D.npy')
        self.erase_path = video_feat_path + '_erTrue'
        if action_type == 'TES':
            self.score_range = 45
            args.score_range = 45
        else:
            self.score_range = 40
            args.score_range = 40
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            labels.append([line[0], float(line[idx[action_type]])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(self.video_path, allow_pickle=True).item()[self.labels[idx][0]]
        audio_feat = np.load(self.audio_path, allow_pickle=True).item()[self.labels[idx][0]]
        flow_feat = np.load(self.flow_path, allow_pickle=True).item()[self.labels[idx][0]]
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range