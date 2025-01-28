from torch.utils.data import Dataset
import torch

class hrgldd_dataset(Dataset): 
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
      
    def __getitem__(self, index):
        img = self.x[index]
        label = self.y[index]

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        return img, label
