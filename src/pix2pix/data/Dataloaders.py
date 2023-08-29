from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

class Pix2Pix(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(),])
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
        self.transform = transform
        self.files = os.listdir(self.root_dir)
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        img = img.convert('RGB')
        # mask is the first 256*256 pixels
        mask = img.crop((0, 0, 256, 256)).convert('L')
        #invert mask colors
        mask = Image.eval(mask, lambda x: 255-x)
        # convert to numpy array
        mask = np.array(mask)
        mask[mask < 50] = 0
        mask[mask >= 50] = 255
        mask = Image.fromarray(mask)
        img = img.crop((256, 0, 512, 256))
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        #return as a dictionary
        return {'image': img, 'mask': mask}

def create_train_loader(root_dir, batch_size):
    dataset = Pix2Pix(root_dir, is_train=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def create_test_loader(root_dir, batch_size):
    dataset = Pix2Pix(root_dir, is_train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)