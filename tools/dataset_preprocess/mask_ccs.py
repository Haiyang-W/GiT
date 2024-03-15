import json
import os
img_dir = "/u/sshi/workspace_ptmp/workspace_hy2/VisionFoundation/data/ccs/images"
valid_num = 0
invalid_num = 0
import cv2
from skimage import io
from datasets.utils.file_utils import get_datasets_user_agent
import urllib.request

USER_AGENT = get_datasets_user_agent()
with open("/u/sshi/workspace_ptmp/workspace_hy2/BLIP/dataset/ccs_synthetic_filtered.json","r") as f:
    anno = json.load(f)
    mask_anno = []
    for i,an in enumerate(anno):
        img_path = os.path.join(img_dir,f'{i}.jpg')
        if os.path.exists(img_path):
            an['exist'] = True
            valid_num += 1
            if i < 100000:
                if cv2.imread(img_path) is None:
                    os.remove(img_path)
                    print(f"remove {img_path}")
                    an['exist'] = False
        else:
            print(f'not exist {i}')
            an['exist'] = False
            invalid_num += 1
        mask_anno.append(an)
print(f'valid num {valid_num}, invalid_num {invalid_num}')
with open("/u/sshi/workspace_ptmp/workspace_hy2/VisionFoundation/data/ccs/ccs_synthetic_filtered_mask_12m.json","w") as f:
    json.dump(mask_anno,f)