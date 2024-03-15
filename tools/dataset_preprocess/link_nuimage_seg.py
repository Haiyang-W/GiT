import cv2 
import json
import os 
from tqdm import tqdm

with open("/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/annotations/nuimages_v1.0-val.json","r") as f:
    train_info = json.load(f)
img_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/'
seg_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/annotations/semantic_masks/'
target_img_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages_seg/images/validation/'
target_seg_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages_seg/annotations/validation/'
os.makedirs(target_img_dir,exist_ok=True)
os.makedirs(target_seg_dir,exist_ok=True)
for img_info in tqdm(train_info['images']):
    img_path = os.path.join(img_dir,img_info['file_name'])
    target_path = os.path.join(target_img_dir,os.path.basename(img_path))
    os.system(f'ln -s {img_path} {target_img_dir}')
    seg_path = os.path.join(seg_dir,img_info['file_name'].replace('jpg','png'))
    target_seg_path = os.path.join(target_seg_dir,os.path.basename(img_path))
    os.system(f'ln -s {seg_path} {target_seg_dir}')

with open("/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/annotations/nuimages_v1.0-train.json","r") as f:
    train_info = json.load(f)
img_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/'
seg_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages/annotations/semantic_masks/'
target_img_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages_seg/images/training/'
target_seg_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/nuimages_seg/annotations/training/'
os.makedirs(target_img_dir,exist_ok=True)
os.makedirs(target_seg_dir,exist_ok=True)
for img_info in tqdm(train_info['images']):
    img_path = os.path.join(img_dir,img_info['file_name'])
    target_path = os.path.join(target_img_dir,os.path.basename(img_path))
    os.system(f'ln -s {img_path} {target_img_dir}')
    seg_path = os.path.join(seg_dir,img_info['file_name'].replace('jpg','png'))
    target_seg_path = os.path.join(target_seg_dir,os.path.basename(img_path))
    os.system(f'ln -s {seg_path} {target_seg_dir}')