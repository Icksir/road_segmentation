import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchinfo import summary
import segmentation_models_pytorch as smp
from torchvision import transforms
import os
from pathlib import Path
from tqdm.auto import tqdm
import re
import warnings
warnings.filterwarnings("ignore")
import classes
import utils

# -------------

PATH = 'images'
EXTENSION_ARCHIVO = '.png'
SEGM = 'road_surface' # or 'road_surface'/'marking'
RESIZE = (512, 512)
BATCH_SIZE = 1
NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -------------

torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

IMAGE_PATH_LIST = [img for img in os.listdir(PATH) if img.endswith(EXTENSION_ARCHIVO)]
IMAGE_PATH_LIST = utils.sorted_alphanumeric(IMAGE_PATH_LIST)
IMAGE_PATH_LIST = ["images/"+img for img in IMAGE_PATH_LIST]

images_paths = [None] * len(IMAGE_PATH_LIST)

for i,(img_path) in enumerate(zip(IMAGE_PATH_LIST)):
    images_paths[i] = img_path[0]
    
data = pd.DataFrame({'Image':images_paths})

image_transforms = transforms.Compose([transforms.Resize(RESIZE), 
                                       transforms.ToTensor()])
mask_transforms = transforms.Compose([transforms.Resize(RESIZE)])

color2id = {(184, 61, 245): 0, # #b83df5: backgroud
            (255, 53, 94):1, # #ff355e: road_sign
            (255, 204, 51):2, # #ffcc33: car
            (221, 255, 51):3, # #ddff33: marking
            (61,61, 245):4} # #3d3df5: road_surface

id2color = {0: (184, 61, 245, 0), # background
            1: (255, 53, 94, 0), # road_sign
            2: (255, 204, 51, 0), # car
            3: (221, 255, 51, 150), # marking
            4: (61,61, 245, 150)} # road_surface

test_dataset = classes.CustomDataset(data, color2id, image_transforms, mask_transforms)

print(data)

test_dataloader = DataLoader(dataset = test_dataset, 
                             batch_size = BATCH_SIZE, 
                             shuffle = False)

pred_mask_test = utils.predictions(test_dataloader, "best_model.pth")

total_mask_output = []
total_mask_marking = []
total_mask_road_surface = []

IMAGE_TEST = []
MASK_TEST = []

for img in test_dataloader:
    IMAGE_TEST.append(img)

IMAGE_TEST = torch.cat(IMAGE_TEST)

total_mask_output = []

for i,mask_pred in enumerate(pred_mask_test):
    
    # We extract the height and width of the mask.
    height,width = mask_pred.shape
    
    mask_zeros = torch.zeros(size = (height, width, 4), dtype = torch.uint8)
    
    for h in range(height):
        for w in range(width):

            if SEGM == 'marking' and mask_pred[h,w] == 3:
                mask_zeros[h,w,:] = torch.tensor(id2color[3])

            if SEGM == 'road_surface' and mask_pred[h,w] == 4:
                mask_zeros[h,w,:] = torch.tensor(id2color[4])

    total_mask_output.append(mask_zeros)

for i,mask_out in enumerate(total_mask_output):
    image = cv2.imread(IMAGE_PATH_LIST[i])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    mask = mask_out.cpu().detach().numpy()
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (image.shape[1],image.shape[0]))

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image[mask==255] = (24,41,88,75)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/' + str(i) +".png", image)