"""LFW dataloading."""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        self.image_paths = glob.glob(path_to_folder + "/**/*.jpg")
        self.transform = transform

    def __len__(self):
        """Return length of dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        img = Image.open(self.image_paths[index])
        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="../../../lfw-funneled/lfw_funneled", type=str)
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-num_workers", default=0, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=10, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        for imgs in dataloader:
            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()
            break

    if args.get_timing:
        # lets do some repetitions
        worker_arr = [0,2,4,6,8]
        runs = 5
        
        res = np.zeros((len(worker_arr),runs))
        for i,workers in tqdm(enumerate(worker_arr)):
            # setup new dataloader
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers)
            # run through 
            for j in range(runs):
                start = time.time()
                for batch_idx, _batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res[i,j] = end - start

        plt.errorbar(x=worker_arr, y=np.mean(res,axis=1), yerr=np.std(res,axis=1))
        plt.xlabel("number of workers")
        plt.ylabel("Time [s]")
        plt.show()