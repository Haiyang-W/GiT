### Training
#### Single Task 
Detection

```shell
bash tools/dist_train.sh configs/GiT/single_detection_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Semantic Segmentation

```shell
bash tools/dist_train.sh configs/GiT/single_semanticseg_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Instance Segmentation

```shell
bash tools/dist_train.sh configs/GiT/single_instanceseg_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Image Caption

```shell
bash tools/dist_train.sh configs/GiT/single_caption_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Visual Grounding

```shell
bash tools/dist_train.sh configs/GiT/single_visualgrounding_base.py ${GPU_NUM} --work-dir ${work_dir}
```

#### Multi Task 

GiT-B

```shell
bash tools/dist_train.sh configs/GiT/multi_fivetask_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

GiT-L

```shell
bash tools/dist_train.sh configs/GiT/multi_fivetask_large.py ${GPU_NUM} --work-dir ${work_dir}
```

GiT-H

```shell
bash tools/dist_train.sh configs/GiT/multi_fivetask_huge.py ${GPU_NUM} --work-dir ${work_dir}
```
#### Universal Training

GiT-B

```shell
bash tools/dist_train.sh configs/GiT/universal_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

GiT-L

```shell
bash tools/dist_train.sh configs/GiT/universal_large.py ${GPU_NUM} --work-dir ${work_dir}
```

GiT-H

```shell
bash tools/dist_train.sh configs/GiT/universal_huge.py ${GPU_NUM} --work-dir ${work_dir}
```
### Testing

#### Single Task 
Detection

```shell
bash tools/dist_test.sh configs/GiT/single_detection_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Semantic Segmentation

```shell
bash tools/dist_test.sh configs/GiT/single_semanticseg_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Instance Segmentation

```shell
bash tools/dist_test.sh configs/GiT/single_instanceseg_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Image Caption

```shell
bash tools/dist_test.sh configs/GiT/single_caption_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Visual Grounding

```shell
bash tools/dist_test.sh configs/GiT/single_visualgrounding_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

#### Multi Task 

GiT-B

```shell
bash tools/dist_test.sh configs/GiT/multi_fivetask_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

GiT-L

```shell
bash tools/dist_test.sh configs/GiT/multi_fivetask_large.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

GiT-H

```shell
bash tools/dist_test.sh configs/GiT/multi_fivetask_huge.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```
#### Zero-shot
Please download universal pretrain weight from [huggingface](https://huggingface.co/kanashi6/GiT/tree/main) and organize files as follows:
```
GiT
|──universal_base.pth
|——universal_large.pth
|——universal_huge.pth

GiT-B

```shell
bash tools/dist_test.sh configs/GiT/zero-shot/zero_shot_cityscapes_det_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

GiT-L

```shell
bash tools/dist_test.sh configs/GiT/zero-shot/zero_shot_cityscapes_det_large.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

GiT-H

```shell
bash tools/dist_test.sh configs/GiT/zero-shot/zero_shot_cityscapes_det_huge.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```
#### Few-shot
Please download universal pretrain weight from [huggingface](https://huggingface.co/kanashi6/GiT/tree/main) and organize files as follows:
```
GiT
|──universal_base.pth
|——universal_large.pth
|——universal_huge.pth

```
GiT-B

```shell
bash tools/dist_train.sh configs/GiT/few-shot/few_shot_drive_det_base.py ${GPU_NUM} --work-dir ${work_dir}
```

GiT-L

```shell
bash tools/dist_train.sh configs/GiT/few-shot/few_shot_drive_det_large.py ${GPU_NUM} --work-dir ${work_dir}
```

GiT-H

```shell
bash tools/dist_train.sh configs/GiT/few-shot/few_shot_drive_det_huge.py ${GPU_NUM} --work-dir ${work_dir}
```