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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predictions(test_dataloader:torch.utils.data.DataLoader, best_model:str):
    
    checkpoint = torch.load(best_model)
    loaded_model = smp.Unet(encoder_name = "resnet34", encoder_weights = None, classes = 5)
    loaded_model.load_state_dict(checkpoint)
    loaded_model.to(device = DEVICE)
    loaded_model.eval()
    pred_mask_test = []

    with torch.inference_mode():
        for X in tqdm(test_dataloader):
            X = X.to(device = DEVICE, dtype = torch.float32)
            logit_mask = loaded_model(X)
            prob_mask = logit_mask.softmax(dim = 1)
            pred_mask = prob_mask.argmax(dim = 1)
            pred_mask_test.append(pred_mask.detach().cpu())

    pred_mask_test = torch.cat(pred_mask_test)

    return pred_mask_test

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def mapping_color(img:Image, color2id:dict):
    
    image = np.array(img)
    
    height,width,_ = image.shape
    output_matrix = np.full(shape = (height, width), fill_value = -1, dtype = np.int32)
    
    for h in range(height):
        for w in range(width):
            color_pixel = tuple(image[h,w,:])
            
            if color_pixel in color2id:
                output_matrix[h,w] = color2id[color_pixel]
            
    
    return output_matrix

def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, 
               loss_fn:smp.losses, optimizer:torch.optim.Optimizer):
    
    model.train()
    
    train_loss = 0.
    train_iou = 0.
    
    for batch,(X,y) in enumerate(dataloader):
        X = X.to(device = DEVICE, dtype = torch.float32)
        y = y.to(device = DEVICE, dtype = torch.long)
        
        optimizer.zero_grad()
        
        pred_logit = model(X)
        loss = loss_fn(pred_logit, y)
        train_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred_prob = pred_logit.softmax(dim = 1)
        pred_class = pred_prob.argmax(dim = 1)
        
        tp,fp,fn,tn = smp.metrics.get_stats(output = pred_class.detach().cpu().long(), 
                                            target = y.cpu(), 
                                            mode = "multiclass", 
                                            ignore_index = -1, 
                                            num_classes = 5)
        
        train_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction = "micro")
        
    train_loss = train_loss / len(dataloader)
    train_iou = train_iou / len(dataloader)
    
    return train_loss, train_iou

def val_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, 
             loss_fn:smp.losses):
    
    model.eval()
    
    val_loss = 0.
    val_iou = 0.
    
    with torch.inference_mode():
    
        for batch,(X,y) in enumerate(dataloader):
            X = X.to(device = DEVICE, dtype = torch.float32)
            y = y.to(device = DEVICE, dtype = torch.long)

            pred_logit = model(X)
            loss = loss_fn(pred_logit, y)
            val_loss = loss.item()

            pred_prob = pred_logit.softmax(dim = 1)
            pred_class = pred_prob.argmax(dim = 1)

            tp,fp,fn,tn = smp.metrics.get_stats(output = pred_class.detach().cpu().long(), 
                                                target = y.cpu(), 
                                                mode = "multiclass", 
                                                ignore_index = -1, 
                                                num_classes = 5)

            val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction = "micro")
        
        
    val_loss = val_loss / len(dataloader)
    val_iou = val_iou / len(dataloader)
    
    return val_loss, val_iou

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, 
          early_stopping, epochs:int = 10):
    
    results = {'train_loss':[], 'train_iou':[], 'val_loss':[], 'val_iou':[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_iou = train_step(model = model, 
                                           dataloader = train_dataloader, 
                                           loss_fn = loss_fn, 
                                           optimizer = optimizer)
        
        val_loss, val_iou = val_step(model = model, 
                                     dataloader = val_dataloader, 
                                     loss_fn = loss_fn)
        
        print(f'Epoch: {epoch + 1} | ', 
              f'Train Loss: {train_loss:.4f} | ', 
              f'Train IOU: {train_iou:.4f} | ', 
              f'Val Loss: {val_loss:.4f} | ', 
              f'Val IOU: {val_iou:.4f}')
        
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop == True:
            print("Early Stopping!!")
            break
            
        results['train_loss'].append(train_loss)
        results['train_iou'].append(train_iou)
        results['val_loss'].append(val_loss)
        results['val_iou'].append(val_iou)
        
    return results

def loss_and_metric_plot(results:dict):
    training_loss = results['train_loss']
    valid_loss = results['val_loss']
    
    training_iou = results['train_iou']
    valid_iou = results['val_iou']
    
    fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (9,4))
    axes = axes.flat
    
    axes[0].plot(range(len(training_loss)), training_loss)
    axes[0].plot(range(len(valid_loss)), valid_loss)
    axes[0].set_xlabel("Epoch", fontsize = 12, fontweight = "bold", color = "black")
    axes[0].set_ylabel("Loss", fontsize = 12, fontweight = "bold", color = "black")
    axes[0].set_title("Dice Loss", fontsize = 14, fontweight = "bold", color = "blue")
    
    axes[1].plot(range(len(training_iou)), training_iou)
    axes[1].plot(range(len(valid_iou)), valid_iou)
    axes[1].set_xlabel("Epoch", fontsize = 12, fontweight = "bold", color = "black")
    axes[1].set_ylabel("score", fontsize = 12, fontweight = "bold", color = "black")
    axes[1].set_title("IOU", fontsize = 14, fontweight = "bold", color = "red")
    
    fig.tight_layout()
    fig.show()