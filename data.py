import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, stay_df, poi2vec, agent_begin_idx, agent_end_idx):
        super().__init__()
        self.stay_df = stay_df
        self.poi2vec = poi2vec
        self.agent_begin_idx = agent_begin_idx
        self.agent_end_idx = agent_end_idx

    def __len__(self):
        return len(self.agent_begin_idx)
    
    def __getitem__(self, idx):
        begin = self.agent_begin_idx.iloc[idx]
        end = self.agent_end_idx.iloc[idx]
        stay = self.stay_df.iloc[begin:end+1]
        poi_vec = self.poi2vec[stay['poi_id'].values]
        return poi_vec

def data_collator(data):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
