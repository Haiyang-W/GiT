import os

suffix = [str(i) for i in range(10)]
suffix.extend(['a','b','c','d','e','f'])
stage = 'test'
save_dir = '~/workspace_ptmp/workspace_hy/dataset/OpenImages'
for idx in suffix:
    d_path = f'https://storage.googleapis.com/openimages/v5/{stage}-masks/{stage}-masks-{idx}.zip'
    os.system(f'wget {d_path} -O {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
    os.system(f'unzip {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip -d  {save_dir}/segmentation/{stage}/')
    os.system(f'rm {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
stage = 'validation'
for idx in suffix:
    d_path = f'https://storage.googleapis.com/openimages/v5/{stage}-masks/{stage}-masks-{idx}.zip'
    os.system(f'wget {d_path} -O {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
    os.system(f'unzip {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip -d {save_dir}/segmentation/{stage}/')
    os.system(f'rm {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
stage = 'train'
for idx in suffix:
    d_path = f'https://storage.googleapis.com/openimages/v5/{stage}-masks/{stage}-masks-{idx}.zip'
    os.system(f'wget {d_path} -O {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
    os.system(f'unzip {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip -d {save_dir}/segmentation/{stage}/')
    os.system(f'rm {save_dir}/segmentation/{stage}/{stage}-masks-{idx}.zip')
