import clip
from search_dataset import ClipSearchDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch

img_dir = '/home/wangyuxi/codes/SCGNet/datasets/imagenet/images/val'
save_dir = 'results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
dataset = ClipSearchDataset(img_dir = img_dir, preprocess = preprocess)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

img_path_list, embedding_list = [], []
for img, img_path in tqdm(dataloader):
    with torch.no_grad():
        features = model.encode_image(img.to(device))
        features /= features.norm(dim=-1, keepdim=True)
        embedding_list.extend(features.detach().cpu().numpy())
        img_path_list.extend(img_path)

result = {'img_path': img_path_list, 'embedding': embedding_list}
with open(f'{os.path.join(save_dir, "results.pkl")}', 'wb') as f:
    pickle.dump(result, f, protocol=4)
    











