import json
import ipdb
import os
with open("/u/sshi/workspace_ptmp/workspace_hy/dataset/test_download/vg_caption.json") as f:
    content = json.load(f)
    new_annos = []
    print(len(content))
    exit()
    # 768536
    for ann in content:
        img_path = ann['image']
        img_name = img_path.split('/')[-1]
        new_img_path = os.path.join("images",img_name)
        ann['image'] = new_img_path
        new_annos.append(ann)
with open("/u/sshi/workspace_ptmp/workspace_hy2/VisionFoundation/data/vg/vg.json","w") as f:
    json.dump(new_annos,f)
