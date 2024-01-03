import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MnistData(Dataset):
    def __init__(self, datafiles: list, labelfiles: list, data_per_file: int, points: int):
        self.n_files = len(datafiles)
        self.data_per_file = data_per_file
        
        self.data = torch.randn(self.n_files * self.data_per_file, points)
        self.labels = torch.randn(self.n_files * self.data_per_file).long()
        
        # load data 
        counter = 0
        for datafile,lblfile in zip(datafiles,labelfiles):
            tmp_data = torch.load(datafile)
            tmp_lbl = torch.load(lblfile)
            for img,lbl in zip(tmp_data,tmp_lbl):
                self.data[counter] = img.flatten()
                self.labels[counter] = lbl
                counter += 1 
        
        # self.labels = F.one_hot(self.labels.long(), 10)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def mnist(datafolder, batch_size):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_files = [f"{datafolder}/train_images_{i}.pt" for i in range(6)]
    tr_label_files = [f"{datafolder}/train_target_{i}.pt" for i in range(6)]
    test_files = [f"{datafolder}/test_images.pt"]
    ts_label_files = [f"{datafolder}/test_target.pt"]
    
    traindata = MnistData(train_files, tr_label_files, data_per_file=5000, points=28*28)
    testdata = MnistData(test_files, ts_label_files, data_per_file=5000, points=28*28)
    
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


# train_files = [f"dtu_mlops/data/corruptmnist/train_images_{i}.pt" for i in range(6)]
# tr_label_files = [f"dtu_mlops/data/corruptmnist/train_target_{i}.pt" for i in range(6)]
# MnistData(train_files, tr_label_files, data_per_file=5000, points=28*28)