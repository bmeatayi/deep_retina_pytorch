import os

import numpy as np
import torch.utils.data as data

import math


class SpikeDataset	(data.Dataset):
    def __init__(self, spike_rate_path, stim_path, cell_idx=None, win_len=5, n_split=1, fr_st_end=None, n_cnt_hist=0):
        """
        Spike count dataset

        :param spike_rate_path: Path to the folder containing spike rate files
        :param stim_path: Path to the folder containing stimulation file (stim.npy)
        :param cell_idx: List of cells to be included in the dataset (None=> selecting all cells)
        :param win_len: Number of frames in each data chunk (Before upsampling)
        :param n_split: Upsampling factor
        :param fr_st_end: Start and end frame of the dataset (if None=> uses all frames)
        :param n_cnt_hist: Number of time bins returned in addition to the last spike count
        """
        super(SpikeDataset, self).__init__()

        self.spike_rate_path = spike_rate_path
        self.stim_path = stim_path
        self.n_split = n_split
        self.cell_idx = cell_idx
        self.win_len = win_len
        self.cell_list = sorted(os.listdir(self.spike_rate_path), key=lambda x: int(x.partition('_')[2].partition('.')[0]))
        self.nBins = np.load(spike_rate_path + self.cell_list[0]).shape[0]
        self.n_cnt_hist = n_cnt_hist
        if cell_idx is None:
            cell_idx = list(range(0, len(self.cell_list)))

        self.spike_rate = np.array([]).reshape((self.nBins, 0))

        for idx in cell_idx:
            temp = np.load(spike_rate_path+self.cell_list[idx])
            temp = temp.reshape((self.spike_rate.shape[0], 1))
            self.spike_rate = np.concatenate((self.spike_rate, temp), axis=1)

        self.stim = np.load(self.stim_path + 'stim.npy')

        if fr_st_end is not None:
            self.spike_rate = self.spike_rate[fr_st_end[0]: fr_st_end[1], :]
            self.stim = self.stim[fr_st_end[0]: fr_st_end[1], :, :]

        self.nFrames = self.stim.shape[0]
        self.length = int((self.nFrames - self.win_len) * self.n_split) + 1
        print("Shape of spike rate array:", self.spike_rate.shape)
        print("Shape of stim:", self.stim.shape)

    def __len__(self):
        """Returns length of dataset"""
        return self.length

    def __getitem__(self, idx):
        """
        Returns idx_th chunk of stimulus and corresponding response
        :param idx: index of data chunk
        :return: stim: Stimuli matrix
                cnt: Spike counts
        """
        cnt_idx = idx + self.win_len * self.n_split - 1
        cnt = np.clip(self.spike_rate[(cnt_idx-self.n_cnt_hist):cnt_idx+1, :], a_min=0, a_max=None)
        rem = (idx % self.n_split)
        lf = math.floor(idx / self.n_split)
        stim = self.stim[lf:(lf + self.win_len + 1), :, :].repeat(self.n_split, axis=0)
        stim = stim[rem:(self.n_split * self.win_len) + rem, :, :]
        return stim, cnt