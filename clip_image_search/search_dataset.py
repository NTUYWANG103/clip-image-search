from torch.utils.data import Dataset
import os
from PIL import Image
import clip


class ClipSearchDataset(Dataset):
    def __init__(self, img_dir,  img_ext_list = ['.jpg', '.png', '.jpeg', '.tiff'], preprocess = None):    
        self.preprocess = preprocess
        self.img_path_list = []
        for root, dirs, files in os.walk(img_dir):
            self.img_path_list.extend(os.path.join(root, file) for file in files if os.path.splitext(file)[1].lower() in img_ext_list)

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.processor(img)
        return img, img_path
    

