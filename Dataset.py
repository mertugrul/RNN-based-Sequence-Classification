import torch
from torch.utils import data

class Dataset(data.Dataset):
  def __init__(self, fold_x, fold_y):
      self.fold_x = fold_x
      self.fold_y = fold_y
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def __len__(self):
        #return self.tensor_x.size(0)
        return self.fold_x.shape[0]

  def __getitem__(self, index):
        x = torch.from_numpy(self.fold_x[index]).to(self.device, dtype=torch.float)
        y = torch.tensor(self.fold_y[index]).to(self.device, dtype=torch.long)
        return x, y