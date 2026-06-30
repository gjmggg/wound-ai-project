#!/usr/bin/env python
# coding: utf-8

# **Suupervised training** <br>
# Labels 1 and 3 of some images have been augmented, so that the dataset contains more images that have labels 1 and 3.

# In[1]:




# In[ ]:


import sys

#sys.path.append('/content/drive/MyDrive/Wound_tissue_segmentation')
#sys.path.append('/content/drive/MyDrive/Wound_tissue_segmentation/utils')
#sys.path.append('/content/drive/MyDrive/Wound_tissue_segmentation/wound_lib')


# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses, base
import random
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F

#get_ipython().run_line_magic('matplotlib', 'inline')


# ### Dataloader

# In[ ]:


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            list_IDs,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            to_categorical:bool=False,
            resize=(False, (256, 256)), # To resize, the first value has to be True
            n_classes:int=6,
            default_img=None,
            default_mask=None,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.to_categorical = to_categorical
        self.resize = resize
        self.n_classes = n_classes
        self.default_img = default_img
        self.default_mask = default_mask

    def __getitem__(self, i):
        try:
              # read data
              image = cv2.imread(self.images_fps[i])
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              mask = cv2.imread(self.masks_fps[i], 0)     # ----------------- pay attention ------------------ #
        except Exception as e:
            print("********** Error occured loading default image and mask. *********")
            image = self.default_img
            mask = self.default_mask

        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.resize[1], interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=-1)  # adding channel axis # ----------------- pay attention ------------------ #

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.to_categorical:
            mask = torch.from_numpy(mask)
            mask = F.one_hot(mask.long(), num_classes=self.n_classes)
            mask = mask.type(torch.float32)
            mask = mask.numpy()
            mask = np.squeeze(mask)

            mask = np.moveaxis(mask, -1, 0)
            return image, torch.from_numpy(mask).float()
        else:
    # 确保返回LongTensor
            mask_tensor = torch.from_numpy(mask)
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze()
                return image, mask_tensor.long() 

    def __len__(self):
        return len(self.ids)


# ### Augmentation

# In[ ]:


def get_training_augmentation():
    train_transform = [

        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ],
            p=0.8,
        ),

        albu.OneOf(
            [
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0), # scale only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0), # rotate only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0), # shift only
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0), # affine transform
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Perspective(p=0.2),
                albu.GaussNoise(p=0.2),
                albu.Sharpen(p=0.2),
                albu.Blur(blur_limit=3, p=0.2),
                albu.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.CLAHE(p=0.25),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.RandomGamma(p=0.25),
                albu.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),

    ]

    return albu.Compose(train_transform, p=0.9) # 90% augmentation probability


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# ### Parameters

# In[ ]:


# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 16
n_classes = 4
ACTIVATION = 'softmax2d'   # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001 # learning rate
EPOCHS = 500
WEIGHT_DECAY = 1e-5
SAVE_WEIGHTS_ONLY = True
RESIZE = (False, (256,256)) # if resize needed
TO_CATEGORICAL = False
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False

PERIOD = 10 # periodically save checkpoints
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
RETRAIN = False

# For early stopping
EARLY_STOP = True # True to activate early stopping
PATIENCE = 50 # for early stopping

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ### Helper function: save a model

# In[ ]:


def save(model_path, epoch, model_state_dict, optimizer_state_dict):

    state = {
        'epoch': epoch + 1,
        'state_dict': deepcopy(model_state_dict),
        'optimizer': deepcopy(optimizer_state_dict),
        }

    torch.save(state, model_path)


# ### Loss, optimizer, metrics, and callbacks

# In[ ]:


# Loss function
 
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from segmentation_models_pytorch.utils import base
 
dice_loss = DiceLoss(mode='multiclass')
focal_loss = FocalLoss(mode='multiclass')
dice_loss.__name__ = "DiceLoss"   
focal_loss.__name__ = "FocalLoss"  
total_loss = base.SumOfLosses(dice_loss, focal_loss)

# Metrics
# Loss, optimizer, metrics, and callbacks
# Metrics
# Metrics
metrics = []  


# 
# ### Model Run

# In[ ]:


# Create a function to read names from a text file, and add extensions
def read_names(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()

  names = [name.strip("\n") for name in names] # remove newline

  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names


# ### Training

# In[ ]:


save_dir_pred_root = '/root/autodl-tmp/DFUTissueSegNet-main/predictions'
os.makedirs(save_dir_pred_root, exist_ok = True)

aux_params=dict(
    classes=n_classes,
    activation=ACTIVATION,
    dropout=0.1, # dropout ratio, default is None
)

# create segmentation model with pretrained encoder
# ============ 模型增强配置 ============
USE_DEEP_SUPERVISION = False   
DECODER_ATTENTION = 'scse'  
 
if DECODER_ATTENTION:
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=n_classes,
        activation=ACTIVATION,
        decoder_attention_type=DECODER_ATTENTION,
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),   
        encoder_depth=5,   
    )
    print(f"✅ 使用增强模型: {ENCODER} with {DECODER_ATTENTION} attention")
else:
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=n_classes,
        activation=ACTIVATION,
    )
    print("✅ 使用基础模型")
# ============ 模型增强配置结束 ============

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                              factor=0.1,
                              mode='min',
                              patience=10,
                              min_lr=0.00001,
                              )

seed = random.randint(0, 5000)

print(f'seed: {seed}')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

x_train_dir = x_valid_dir = '/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled/Padded/Images/TrainVal'
y_train_dir = y_valid_dir = '/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled/Padded/Annotations/TrainVal'

x_test_dir = '/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled/Padded/Images/Test'
y_test_dir = '/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled/Padded/Annotations/Test'

# Read train, test, and val names
dir_txt = '/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled'
list_IDs_train = read_names(os.path.join(dir_txt, 'labeled_train_names.txt'), ext='.png')
list_IDs_val = read_names(os.path.join(dir_txt, 'labeled_val_names.txt'), ext='.png')
list_IDs_test = read_names(os.path.join(dir_txt, 'test_names.txt'), ext='.png')

random.seed(seed) # seed for random number generator
random.shuffle(list_IDs_train) # shuffle train names

print('No. of training images: ', len(list_IDs_train))
print('No. of validation images: ', len(list_IDs_val))
print('No. of test images: ', len(list_IDs_test))

# Create a unique model name
model_name = BASE_MODEL + '_padded_aug_' + ENCODER + '_sup_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(model_name)

# Default images
DEFAULT_IMG_TRAIN = cv2.imread(os.path.join(x_train_dir, list_IDs_train[0]))[:,:,::-1]
DEFAULT_MASK_TRAIN = cv2.imread(os.path.join(y_train_dir, list_IDs_train[0]), 0)
DEFAULT_IMG_VAL = cv2.imread(os.path.join(x_valid_dir, list_IDs_val[0]))[:,:,::-1]
DEFAULT_MASK_VAL = cv2.imread(os.path.join(y_valid_dir, list_IDs_val[0]), 0)

# Checkpoint directory
checkpoint_loc = '/root/autodl-tmp/DFUTissueSegNet-main/checkpoints/' + model_name

# Create checkpoint directory if does not exist
if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

# if SAVE_BEST_MODEL_ONLY: checkpoint_path = os.path.join(checkpoint_loc, 'best_model.pth')
# else: checkpoint_path = os.path.join(checkpoint_loc, "cp-{epoch:04d}.pth")

# Dataloader ===================================================================
train_dataset = Dataset(
    list_IDs_train,
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    to_categorical=TO_CATEGORICAL,
    resize=(RESIZE),
    n_classes=n_classes,
    default_img=DEFAULT_IMG_TRAIN,
    default_mask=DEFAULT_MASK_TRAIN,
)

valid_dataset = Dataset(
    list_IDs_val,
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    resize=(RESIZE),
    to_categorical=TO_CATEGORICAL,
    n_classes=n_classes,
    default_img=DEFAULT_IMG_VAL,
    default_mask=DEFAULT_MASK_VAL,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

# create epoch runners =========================================================
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=total_loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=total_loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# Train ========================================================================
# train model for N epochs
best_viou = 0.0
best_vloss = 1_000_000.
save_model = False # Initially start with False
cnt_patience = 0

store_train_loss, store_val_loss = [], []
store_train_iou, store_val_iou = [], []
store_train_dice, store_val_dice = [], []


# Train ========================================================================
# train model for N epochs
best_viou = 0.0
best_vloss = 1_000_000.
save_model = False # Initially start with False
cnt_patience = 0

store_train_loss, store_val_loss = [], []
store_train_iou, store_val_iou = [], []
store_train_dice, store_val_dice = [], []
 
def compute_iou_manual(predictions, targets, n_classes=4):
    """
    手动计算多分类IoU
    predictions: [B, C, H, W]
    targets: [B, H, W]
    """
    import torch
    
    pred_class = torch.argmax(predictions, dim=1)  # [B, H, W]
   
    ious = []
    for cls in range(1, n_classes):   
        pred_cls = (pred_class == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(torch.tensor(0.0).to(predictions.device))
    
    if len(ious) > 0:
        return torch.stack(ious).mean()
    else:
        return torch.tensor(0.0).to(predictions.device)

def compute_dice_manual(predictions, targets, n_classes=4):
    import torch
    
    pred_class = torch.argmax(predictions, dim=1)
    
    dices = []
    for cls in range(1, n_classes):
        pred_cls = (pred_class == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum().float() + target_cls.sum().float()
        
        if total > 0:
            dices.append(2 * intersection / total)
        else:
            dices.append(torch.tensor(0.0).to(predictions.device))
    
    if len(dices) > 0:
        return torch.stack(dices).mean()
    else:
        return torch.tensor(0.0).to(predictions.device)

for epoch in range(EPOCHS):

    print('\nEpoch: {}'.format(epoch))
    model.eval()
    with torch.no_grad():
        train_iou_sum = 0.0
        train_dice_sum = 0.0
        train_batches = 0
        
        for train_images, train_masks in train_loader:
            train_predictions = model(train_images.to(DEVICE))
            train_iou = compute_iou_manual(train_predictions, train_masks.to(DEVICE), n_classes)
            train_dice = compute_dice_manual(train_predictions, train_masks.to(DEVICE), n_classes)
            
            train_iou_sum += train_iou.item()
            train_dice_sum += train_dice.item()
            train_batches += 1
            
            if train_batches >= 2:  
                break
        
        avg_train_iou = train_iou_sum / train_batches if train_batches > 0 else 0.0
        avg_train_dice = train_dice_sum / train_batches if train_batches > 0 else 0.0
    
    with torch.no_grad():
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        val_batches = 0
        
        for val_images, val_masks in valid_loader:
            val_predictions = model(val_images.to(DEVICE))
            val_iou = compute_iou_manual(val_predictions, val_masks.to(DEVICE), n_classes)
            val_dice = compute_dice_manual(val_predictions, val_masks.to(DEVICE), n_classes)
            
            val_iou_sum += val_iou.item()
            val_dice_sum += val_dice.item()
            val_batches += 1
            
            if val_batches >= 1: 
                break
        
        avg_val_iou = val_iou_sum / val_batches if val_batches > 0 else 0.0
        avg_val_dice = val_dice_sum / val_batches if val_batches > 0 else 0.0
    
    print(f'训练IoU: {avg_train_iou:.4f}, 训练Dice: {avg_train_dice:.4f}')
    print(f'验证IoU: {avg_val_iou:.4f}, 验证Dice: {avg_val_dice:.4f}')
    
    # 存储手动计算的Metrics
    store_train_iou.append(avg_train_iou)
    store_val_iou.append(avg_val_iou)
    store_train_dice.append(avg_train_dice)
    store_val_dice.append(avg_val_dice)
    
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Store losses and metrics
    train_loss_key = list(train_logs.keys())[0] # first key is for loss
    val_loss_key = list(valid_logs.keys())[0] # first key is for loss

    # Store losses and metrics
    train_loss_key = list(train_logs.keys())[0] # first key is for loss
    val_loss_key = list(valid_logs.keys())[0] # first key is for loss

    store_train_loss.append(train_logs[train_loss_key])
    store_val_loss.append(valid_logs[val_loss_key])

    # 安全地存储IoU分数
    if "iou_score" in train_logs:
        store_train_iou.append(train_logs["iou_score"])
        store_val_iou.append(valid_logs["iou_score"])
    else:
        # 使用手动计算的值
        store_train_iou.append(avg_train_iou)
        store_val_iou.append(avg_val_iou)
        print(f'使用手动计算的IoU: 训练={avg_train_iou:.4f}, 验证={avg_val_iou:.4f}')

    # 安全地存储Dice分数
    if "fscore" in train_logs:
        store_train_dice.append(train_logs["fscore"])
        store_val_dice.append(valid_logs["fscore"])
    else:
        # 使用手动计算的值
        store_train_dice.append(avg_train_dice)
        store_val_dice.append(avg_val_dice)
        print(f'使用手动计算的Dice: 训练={avg_train_dice:.4f}, 验证={avg_val_dice:.4f}')

    # Track best performance, and save the model's state
    if  best_vloss > valid_logs[val_loss_key]:
        best_vloss = valid_logs[val_loss_key]
        print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
        cnt_patience = 0 # reset patience
        best_model_epoch = epoch
        save_model = True

    # Compare iou score
    elif best_viou < avg_val_iou:   
        best_viou = avg_val_iou   
        print(f'Validation IoU increased. Saving the model at epoch: {epoch:04d}.')
        cnt_patience = 0 # reset patience
        best_model_epoch = epoch
        save_model = True

    else: cnt_patience += 1

    # Learning rate scheduler
    scheduler.step(valid_logs[sorted(valid_logs.keys())[0]]) # monitor validation loss

    # Save the model
    if save_model:
        save(os.path.join(checkpoint_loc, 'best_model' + '.pth'),
            epoch+1, model.state_dict(), optimizer.state_dict())
        save_model = False

    # Early stopping
    if EARLY_STOP and cnt_patience >= PATIENCE:
      print(f"Early stopping at epoch: {epoch:04d}")
      break

    # Periodic checkpoint save
    if not SAVE_BEST_MODEL:
      if (epoch+1) % PERIOD == 0:
        save(os.path.join(checkpoint_loc, f"cp-{epoch+1:04d}.pth"),
            epoch+1, model.state_dict(), optimizer.state_dict())
        print(f'Checkpoint saved for epoch {epoch:04d}')

if not EARLY_STOP and SAVE_LAST_MODEL:
    print('Saving last model')
    save(os.path.join(checkpoint_loc, 'last_model' + '.pth'),
        epoch+1, model.state_dict(), optimizer.state_dict())

print(best_model_epoch)


# ### Plot loss curves

# In[ ]:


# Plot loss curves =============================================================
fig, ax = plt.subplots(1,3, figsize=(12, 3))

ax[0].plot(store_train_loss, 'r')
ax[0].plot(store_val_loss, 'b')
ax[0].set_title('Loss curve')
ax[0].legend(['training', 'validation'])

ax[1].plot(store_train_iou, 'r')
ax[1].plot(store_val_iou, 'b')
ax[1].set_title('IoU curve')
ax[1].legend(['training', 'validation'])

ax[2].plot(store_train_iou, 'r')
ax[2].plot(store_val_iou, 'b')
ax[2].set_title('Dice curve')
ax[2].legend(['training', 'validation'])

fig.tight_layout()

save_fig_dir = "/root/autodl-tmp/DFUTissueSegNet-main"
if not os.path.exists(save_fig_dir): os.makedirs(save_fig_dir)

fig.savefig(os.path.join(save_fig_dir, model_name + '.png'))


# ### Inference

# In[ ]:


# =================================== Inference ================================
# Load model====================================================================
checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Test dataloader ==============================================================
test_dataset = Dataset(
    list_IDs_test,
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    resize=(RESIZE),
    to_categorical=False, # don't convert to onehot now
    n_classes=n_classes,
)

test_dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=6)

# Prediction ===================================================================
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import warnings
warnings.filterwarnings("ignore")

save_pred = True
threshold = 0.5
ep = 1e-6
raw_pred = []

HARD_LINE = True

# Save directory
save_dir_pred = '/root/autodl-tmp/DFUTissueSegNet-main/predictions/' + model_name
save_dir_pred_pal = '/root/autodl-tmp/DFUTissueSegNet-main/predictions_palette/' + model_name
save_dir_pred_pal_cat = '/root/autodl-tmp/DFUTissueSegNet-main/predictions_palette_cat/' + model_name
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)
if not os.path.exists(save_dir_pred_pal): os.makedirs(save_dir_pred_pal)
if not os.path.exists(save_dir_pred_pal_cat): os.makedirs(save_dir_pred_pal_cat)

# Create a dictionary to store metrics
metric = {} # Nested metric format: metric[image_name][label] = [precision, recall, dice, iou]

# fig, ax = plt.subplots(5,2, figsize=(10,15))
iter_test_dataloader = iter(test_dataloader)

palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

stp, stn, sfp, sfn = 0, 0, 0, 0

for i in range(len(list_IDs_test)):

    tp, tn, fp, fn = 0, 0, 0, 0

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    metric[name] = {} # Creating nested dictionary

    # Image-wise mean of metrics
    i_mp, i_mr, i_mdice, i_miou = [], [], [], []

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

    pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu

    # Convert from onehot
    # gt_mask = torch.argmax(gt_mask_, dim=1)
    if TO_CATEGORICAL:
        pr_mask = torch.argmax(pr_mask, dim=1)

    # pr_mask = torch.argmax(pr_mask, dim=1)

    # Move to CPU and convert to numpy
    gt_mask = gt_mask.squeeze().cpu().numpy()
    gt_mask = np.asarray(gt_mask, dtype=np.int64) # convert to integer
    pred = pr_mask.squeeze().cpu().numpy()

# Save raw prediction
if RAW_PREDICTION: 
    raw_pred.append(pred)

# Modify prediction based on threshold
# pred = (pred >= threshold) * 1

# Save prediction as png
if save_pred:
    if pred.ndim > 2:
        if pred.shape[0] <= 4: 
            pred_class = np.argmax(pred, axis=0)
        else:
            pred_class = np.argmax(pred, axis=-1)
    else:
        pred_class = pred

    pred_class = pred_class.astype(np.uint8)
    
    if pred_class.max() <= 3:  
        pred_visual = (pred_class * 63).astype(np.uint8)
    else:
        pred_visual = np.clip(pred_class, 0, 255).astype(np.uint8)
    
    #"Uncomment for non-palette"
    cv2.imwrite(os.path.join(save_dir_pred, list_IDs_test[i]), pred_visual)

    #"Uncomment for palette"
    # Palette original
    pal_gt_mask = np.squeeze(gt_mask).astype(np.uint8)
    pal_gt_mask = Image.fromarray(pal_gt_mask)
    pal_gt_mask = pal_gt_mask.convert("P")
    pal_gt_mask.putpalette(np.array(palette, dtype=np.uint8))

    # Palette prediction
    pal_pred = np.squeeze(pred_class).astype(np.uint8)
    pal_pred = Image.fromarray(pal_pred)
    pal_pred = pal_pred.convert("P")
    pal_pred.putpalette(np.array(palette, dtype=np.uint8))

    pal_pred.save(os.path.join(save_dir_pred_pal, list_IDs_test[i])) # store

    # Concatenate gt and pred side by side
    concat_pals = Image.new("RGB", (pal_gt_mask.width+pal_gt_mask.width, pal_gt_mask.height), "white")
    concat_pals.paste(pal_gt_mask, (0, 0))
    concat_pals.paste(pal_pred, (pal_gt_mask.width, 0))

    concat_pals.save(os.path.join(save_dir_pred_pal_cat, list_IDs_test[i])) # store

if pred.dtype != np.uint8 and pred.dtype != np.int64:
    if pred.ndim == 3 and pred.shape[0] <= 4:   
        pred_int = np.argmax(pred, axis=0).astype(np.uint8)
    elif pred.ndim == 2 and pred.max() <= 1.0:   
        pred_int = (pred >= threshold).astype(np.uint8)
    else:
        pred_int = pred.astype(np.uint8)
else:
    pred_int = pred

# Find labels in gt and prediction
lbl_gt = set(np.unique(gt_mask))
lbl_gt.remove(0) # remove 0. It is background
lbl_pred = set(np.unique(pred_int))
lbl_pred.remove(0) # remove 0. It is background

# All labels
all_lbls = lbl_gt.union(lbl_pred)

# Find labels that are not common in both gt and prediction. For such cases. IoU = 0
diff1 = lbl_gt - lbl_pred
diff2 = lbl_pred - lbl_gt
diffs = diff1.union(diff2) # labels that do not exist in either gt or prediction

# Labels that are in the gt but not in prediction are fn
if len(diff1) > 0:
    for d1 in diff1:
        fn_ = len(np.argwhere(gt_mask == d1))
        fn += fn_
        sfn += fn

# Labels that are in the prediction but not in gt are fp
if len(diff2) > 0:
    for d2 in diff2:
        fp_ = len(np.argwhere(pred_int == d2))
        fp += fp_
        sfp += fp

# Set IoU == 0 for such labels
if not len(diffs) == 0:
    for diff in diffs:
        p, r, dice, iou = 0, 0, 0, 0
        metric[name][str(diff)] = [p, r, dice, iou]
        print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f"%(i+1, name, diff, p, r, dice, iou))
        
    # Find labels that are common in both gt and prediction.
    cmns = lbl_gt.intersection(lbl_pred)

    # Iterate over common labels
    for cmn in cmns:
        gt_idx = np.where(gt_mask == cmn)
        pred_idx = np.where(pred == cmn)

        # Convert to [(x1,y1), (x2,y2), ...]
        gt_lidx, pred_lidx = [], [] # List index

        for i in range(len(gt_idx[0])):
            gt_lidx.append((gt_idx[0][i], gt_idx[1][i]))

        for i in range(len(pred_idx[0])):
            pred_lidx.append((pred_idx[0][i], pred_idx[1][i]))

        # Calculate metrics
        gt_tidx = tuple(gt_lidx) # convert to tuple
        pred_tidx = tuple(pred_lidx) # convert to tuple
        tp_cord = set(gt_tidx).intersection(pred_tidx) # set operation
        fp_cord = set(pred_tidx).difference(gt_tidx) # set operation
        fn_cord = set(gt_tidx).difference(pred_tidx) # set operation

        tp += len(tp_cord)
        fp += len(fp_cord)
        fn += len(fn_cord)

        stp += tp
        sfp += fp
        sfn += fn

        p = (tp/(tp + fp + ep)) * 100
        r = (tp/(tp + fn + ep)) * 100
        dice = (2 * tp / (2 * tp + fp + fn + ep)) * 100
        iou = (tp/(tp + fp + fn + ep)) * 100

        print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f"%(i+1, name, cmn, p, r, dice, iou))

        metric[name][str(cmn)] = [p, r, dice, iou]

        # Keep appending metrics for all labels for the current image
        i_mp.append(p)
        i_mr.append(r)
        i_mdice.append(dice)
        i_miou.append(iou)


# create json object from dictionary
import json
json_write = json.dumps(metric)
f = open(os.path.join(save_dir_pred, "metric.json"), "w")
f.write(json_write)
f.close()

# Data-based evalutation
siou = (stp/(stp + sfp + sfn + ep))*100
sprecision = (stp/(stp + sfp + ep))*100
srecall = (stp/(stp + sfn + ep))*100
sdice = (2 * stp / (2 * stp + sfp + sfn))*100

print('siou:', siou)
print('sprecision:', sprecision)
print('srecall:', srecall)
print('sdice:', sdice)

# Save data-based result in a text file
with open(os.path.join(save_dir_pred, 'result.txt'), 'w') as f:
    print(f'iou = {siou}', file=f)
    print(f'precision = {sprecision}', file=f)
    print(f'recall = {srecall}', file=f)
    print(f'dice = {sdice}', file=f)
    print(f'best model epoch = {best_model_epoch}', file=f)
    print(f'model name = {model_name}', file=f)

