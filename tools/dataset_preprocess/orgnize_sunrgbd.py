import os 
import cv2 
import numpy as np
import shutil
txt_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/sunrgbd'
save_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/sunrgbd_seg'

train_rgb_path = os.path.join(txt_dir,'train_rgb.txt')
train_label_path = os.path.join(txt_dir,'train_label.txt')
test_rgb_path = os.path.join(txt_dir,'test_rgb.txt')
test_label_path = os.path.join(txt_dir,'test_label.txt')
with open(test_rgb_path,'r') as f:
    train_rgb = f.readlines()
    with open(test_label_path,'r') as label_f:
        train_label = label_f.readlines()
        for rgb_path,label_path in zip(train_rgb,train_label):
            rgb_path = os.path.join(txt_dir,rgb_path.strip())
            label_path = os.path.join(txt_dir,label_path.strip())
            rgb = cv2.imread(rgb_path)
            label = np.load(label_path)
            # label = np.expand_dims(label,axis=2).repeat(3,2)
            # import ipdb 
            # ipdb.set_trace()
            assert rgb.shape[0] == label.shape[0]
            basename = os.path.basename(rgb_path)
            os.makedirs(os.path.join(save_dir,'images','validation'),exist_ok=True)
            os.makedirs(os.path.join(save_dir,'annotations','validation'),exist_ok=True)

            save_rgb_path = os.path.join(save_dir,'images','validation',basename)
            save_label_path = os.path.join(save_dir,'annotations','validation',basename.replace('jpg','png'))
            shutil.copy(rgb_path,save_rgb_path)
            cv2.imwrite(save_label_path,label)


with open(train_rgb_path,'r') as f:
    train_rgb = f.readlines()
    with open(train_label_path,'r') as label_f:
        train_label = label_f.readlines()
        for rgb_path,label_path in zip(train_rgb,train_label):
            rgb_path = os.path.join(txt_dir,rgb_path.strip())
            label_path = os.path.join(txt_dir,label_path.strip())
            rgb = cv2.imread(rgb_path)
            label = np.load(label_path)
            # label = np.expand_dims(label,axis=2).repeat(3,2)
            # import ipdb 
            # ipdb.set_trace()
            assert rgb.shape[0] == label.shape[0]
            basename = os.path.basename(rgb_path)
            os.makedirs(os.path.join(save_dir,'images','training'),exist_ok=True)
            os.makedirs(os.path.join(save_dir,'annotations','training'),exist_ok=True)
            save_rgb_path = os.path.join(save_dir,'images','training',basename)
            save_label_path = os.path.join(save_dir,'annotations','training',basename.replace('jpg','png'))
            shutil.copy(rgb_path,save_rgb_path)
            cv2.imwrite(save_label_path,label)