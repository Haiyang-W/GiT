import urllib.request
from PIL import Image
import json
from multiprocessing import  Process
import os
from tqdm import tqdm
import requests
import socket
socket.setdefaulttimeout(1)
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image
import io
USER_AGENT = get_datasets_user_agent()
def down_img(samples,idx):
    split = 20
    save_dir = '/u/sshi/workspace_ptmp/workspace_hy/dataset/ccs/images'
    for i in tqdm(range(len(samples))):
        if i % split == idx:
            url = samples[i]['url']
            save_path = os.path.join(save_dir,f'{i}.jpg')
            if os.path.exists(save_path):
                continue
            try:
                # request = urllib.request.urlopen(url,timeout=1)
                # with open(save_path,"wb") as f:
                #     f.write(request.read())
                request = urllib.request.Request(
                    url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=1) as req:
                    image = Image.open(io.BytesIO(req.read()))
                    image.save(save_path)
            except:
                # if os.path.exists(save_path):
                #     os.remove(save_path)
                # ret = os.system(f'wget {url} -O {save_path} -t 1')
                # if ret != 0:
                print(f'fail to download {url} with id {i}')
data_file = '/u/sshi/workspace_ptmp/workspace_hy2/BLIP/dataset/ccs_synthetic_filtered.json'
with open(data_file,'r') as f:
    samples = json.load(f)
print(samples[0])
process_list = []
for i in range(20):
    p = Process(target=down_img,args=(samples,i))
    p.start()
    process_list.append(p)
for i in process_list:
    p.join()
print("download over")