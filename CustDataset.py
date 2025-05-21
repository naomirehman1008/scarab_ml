import torch
from torch.utils.data import Dataset
import pandas as pd

class CustDataset(Dataset):
    def __init__(self, feather_files, N, stride, feature_list, penalty_weight):
        """
        Args:
            feather_files (list): feathers containing dataset
            N (int): number of fetch targets in a datapoint
            stride (int): step size for fetch targets in a datapoint (1 is contiguous)
            features (list): names of columns in dataset to use as features
        """
        self.N = N * stride
        self.S = stride
        self.penalty_weight = penalty_weight
        data = pd.concat([pd.read_feather(feather_file) for feather_file in feather_files])
        self.length = len(data)
        self.labels = data['off_path'].to_numpy()

        assert(len(self.labels) == self.length)
        self.penalties = data['penalty'].to_numpy()
        self.off_path_lens = data['length_off_path'].to_numpy()
        self.window_ids = data['window_id'].to_numpy()
        self.cycle_penalties = data['cycle_penalty'].to_numpy()
        self.off_path_cycles = data['off_path_cycles'].to_numpy()

        # features are everything else
        self.features = data[feature_list].to_numpy()
        assert self.features.shape[1] == len(feature_list), f"Dataset n features ({self.features.shape[1]}) doesn't match config  features ({len(feature_list)})"
        # use this to control whether penalty is returned with data
        self.mode = 'train'

        # free up some memory
        del(data)

    def __len__(self):
        return self.length
    
    def _set_mode(self, mode: str):
        self.mode = mode
    
    def __getitem__(self, idx):
        if idx < self.N - 1:
            print(f"idx: {idx}, N: {self.N}")
            # SUS
            return self.__getitem__(idx + self.N)
        start_idx = idx - self.N + 1
        end_idx = idx + 1

        # Slice the features and flatten or reshape as needed
        print(self.features.shape)
        print(start_idx)
        print(end_idx)
        print(self.N)
        print(self.S)
        features = self.features[start_idx:end_idx:self.S].flatten()
        print(features.shape)
        label = self.labels[idx]

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        penalty = self.penalties[idx]
        if(self.mode == 'train'):
            return features_tensor, label_tensor, torch.tensor(1 + max((penalty * self.penalty_weight, 0.5)), dtype=torch.float32)
        # return penalty for accuracy
        elif(self.mode == 'test'):
            penalty = self.penalties[idx]
            off_path_len = self.off_path_lens[idx]
            window_id = self.window_ids[idx]
            cycle_penalty = self.cycle_penalties[idx]
            off_path_cycle = self.off_path_cycles[idx]

            return features_tensor, label_tensor, penalty, off_path_len, window_id, cycle_penalty, off_path_cycle
        
        else:
            assert(False), "UH OH, invalid mode for cust data set"