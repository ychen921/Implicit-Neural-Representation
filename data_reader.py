from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class SingleImageDataset(Dataset):
    def __init__(self, img_path):
        self.image = read_image(img_path)
        self.num_channels, self.h, self.w = self.image.shape

    def __len__(self):
        ### TODO: 1 line of code for returning the number of pixels
        return self.h * self.w

    def __getitem__(self, idx):
        ### TODO: 2-3 lines of code for x, y, and pixel values
        x = (idx // self.w) % self.h
        y = idx % self.w
        intensity = np.asarray(self.image)[:, y, x]/255

        return {"x": x, "y": y, "intensity": intensity}