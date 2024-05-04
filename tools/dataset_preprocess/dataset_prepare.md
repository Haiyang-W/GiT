# Training Dataset
```
GiT
|──data
|  |──ade
|  |  |──ADEChallengeData2016
|  |  |  |──annorations
|  |  |  |  |──training & validation
|  |  |  |──images
|  |  |  |  |──training & validation
|  |  |  |──objectInfo150.txt
|  |  |  |──sceneCategories.txt
|  |──coco
|  |  |──annotations
|  |  |  |──*.json
|  |  |──train2017
|  |  |  |──*.jpg
|  |  |──val2017
|  |  |  |──*.jpg
|  |──coco_2014
|  |  |──annotations
|  |  |  |──*.json
|  |  |  |──coco_karpathy_test.json
|  |  |  |──coco_karpathy_train.json
|  |  |  |──coco_karpathy_val_gt.json
|  |  |  |──coco_karpathy_val.json
|  |  |──train2014
|  |  |  |──*.jpg
|  |  |──val2014
|  |  |  |──*.jpg
|  |  |──refcoco
|  |  |  |──*.p
|  |  |──refcoco+
|  |  |  |──*.p
|  |  |──refcocog
|  |  |  |──*.p
|  |──OpenImages
|  |  |──annotations (follow https://github.com/open-mmlab/mmdetection/tree/main/configs/openimages)
|  |  |  |──train-annotations-object-segmentation_sort_resize.csv
|  |  |  |──val-annotations-object-segmentation_sort_resize.csv
|  |  |──OpenImages
|  |  |  |──train
|  |  |  |  |──*.jpg
|  |  |  |──validation
|  |  |  |  |──*.jpg
|  |  |  |──test
|  |  |  |  |──*.jpg
|  |  |──segmentation
|  |  |  |──train
|  |  |  |  |──*.png
|  |  |  |──validation
|  |  |  |  |──*.png
|  |  |  |──test
|  |  |  |  |──*.png

|  |──Objects365
|  |  |──Obj365_v2
|  |  |  |──annotations
|  |  |  |  |──*.jpg
|  |  |  |──train
|  |  |  |  |──patch0
|  |  |  |  |──patch1
             ...
|  |  |  |──val
|  |  |  |  |──patch0
|  |  |  |  |──patch1
             ...
|  |  |  |──annotations
|  |  |  |  |──zhiyuan_objv2_train.json 
|  |  |  |  |──zhiyuan_objv2_val.json
             ...
|  |──lvis_v1
|  |  |──annotations
|  |  |  |──*.json
|  |  |──train2017
|  |  |  |──*.jpg
|  |  |──val2017
|  |  |  |──*.jpg

|  |──coco_stuff164k
|  |  |──annotations
|  |  |  |──train2017
|  |  |  |  |——000000250893_labelTrainIds.png 
|  |  |  |  |——000000250893.png 
|  |  |  |——val2017
|  |  |  |  |——000000229601_labelTrainIds.png
|  |  |  |  |——000000229601.png
|  |  |──images
|  |  |  |──train2017
|  |  |  |  |——*.png 
|  |  |  |——val2017
|  |  |  |  |——*.png 

│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   |   ├── Annotations
│   │   |   |   ├── *.xml
│   │   |   ├── ImageSets
│   │   |   |   ├── Action
│   │   |   |   ├── Layout
│   │   |   |   ├── Main
│   │   |   |   ├── Segmentation
│   │   |   ├── JPEGImages
│   │   |   |   ├── *.jpg
│   │   |   ├── SegmentationClass
│   │   |   |   ├── *.png
│   │   |   ├── SegmentationObject
│   │   |   |   ├── *.png
│   │   ├── VOC2012
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json

|  |──nuimages
|  |  |──annotations
|  |  |  |── nuimages_v1.0-train.json
|  |  |  |── nuimages_v1.0-val.json
|  |  |──calibrated
|  |  |──samples
|  |  |  |── CAM_BACK
|  |  |  |—— CAM_BACK_LEFT
|  |  |  |—— CAM_BACK_RIGHT
|  |  |  |—— CAM_FRONT
|  |  |  |—— CAM_FRONT_LEFT
|  |  |  |—— CAM_FRONT_RIGHT
|  |  |──v1.0-mini
|  |  |──v1.0-test
|  |  |──v1.0-train
|  |  |──v1.0-val

|  |──nuimages_seg
|  |  |──annotations
|  |  |  |── training
|  |  |  |   |── *.png
|  |  |  |── validation
|  |  |──images
|  |  |  |── training
|  |  |  |   |── *.jpg
|  |  |  |── validation

│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json

│   ├── mapillary
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   │   ├── validation
│   │   │   ├── images
|   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons

│   ├── ccs
│   │   ├── images
│   │   │   ├── *.jpg
│   │   ├── ccs_synthetic_filtered_mask_12m.json

│   ├── vg
│   │   ├── images
│   │   │   ├── *.jpg
│   │   ├── vg.json

│   ├── refclef
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   |   ├── images
│   │   ├── refs(unc).p
│   │   ├── instances.json

│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val

│   ├── kitti
│   │   ├── train
│   │   │   ├── image
│   │   │   ├── *.png
│   │   │   ├── coco
│   │   │   │   ├── kitti_coco_format_train.json
│   │   ├── val
│   │   ├── train.txt
│   │   ├── val.txt

│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip

│   ├── SUN RGB-D
│   │   ├── images
│   │   │   ├── training
│   │   │   │   ├── *.jpg
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   │   ├── *.png
│   │   │   ├── validation

│   ├── nocaps
│   │   ├── annotations
│   │   │   ├── nocaps_test_image_info.json
│   │   │   ├── nocaps_val_4500_captions.json
│   │   ├── images
│   │   │   ├── *.jpg

│   ├── loveDA
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val

│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val

│   ├── DRIVE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation

│   ├── WIDERFace
│   │   ├── WIDER_train
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── WIDER_val
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── val.txt
│   │   ├── train.txt

│   ├── DeepFashion
│   │   ├── In-shop
|   │   │   ├── Anno
|   │   │   │   ├── segmentation
|   │   │   │   |   ├── DeepFashion_segmentation_train.json
|   │   │   │   |   ├── DeepFashion_segmentation_query.json
|   │   │   │   |   ├── DeepFashion_segmentation_gallery.json
|   │   │   │   ├── list_bbox_inshop.txt
|   │   │   │   ├── list_description_inshop.json
|   │   │   │   ├── list_item_inshop.txt
|   │   │   │   └── list_landmarks_inshop.txt
|   │   │   ├── Eval
|   │   │   │   └── list_eval_partition.txt
|   │   │   ├── Img
|   │   │   │   ├── img
|   │   │   │   │   ├──XXX.jpg
|   │   │   │   ├── img_highres
|   │   │   │   └── ├──XXX.jpg

```

## COCO 2017
python tools/misc/download_dataset.py --dataset-name coco2017

## ADE20K
mim download mmsegmentation --dataset ade20k

## COCO Caption
python tools/misc/download_dataset.py --dataset-name coco2014
- karpathy download link: https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml

```shell
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json
```

## OpenImages
1. You need to download and extract Open Images dataset.

2. The Open Images dataset does not have image metas (width and height of the image),
   which will be used during training and testing (evaluation). We suggest to get test image metas before
   training/testing by using `tools/misc/get_image_metas.py`.

   **Usage**

   ```shell
   python tools/misc/get_image_metas.py ${CONFIG} \
   --dataset ${DATASET TYPE} \  # train or val or test
   --out ${OUTPUT FILE NAME}
   ```

3. The directory should be like this:

   ```none
   mmdetection
   ├── mmdet
   ├── tools
   ├── configs
   ├── data
   │   ├── OpenImages
   │   │   ├── annotations
   │   │   │   ├── bbox_labels_600_hierarchy.json
   │   │   │   ├── class-descriptions-boxable.csv
   │   │   │   ├── oidv6-train-annotations-bbox.scv
   │   │   │   ├── validation-annotations-bbox.csv
   │   │   │   ├── validation-annotations-human-imagelabels-boxable.csv
   │   │   │   ├── validation-image-metas.pkl      # get from script
   │   │   ├── challenge2019
   │   │   │   ├── challenge-2019-train-detection-bbox.txt
   │   │   │   ├── challenge-2019-validation-detection-bbox.txt
   │   │   │   ├── class_label_tree.np
   │   │   │   ├── class_sample_train.pkl
   │   │   │   ├── challenge-2019-validation-detection-human-imagelabels.csv       # download from official website
   │   │   │   ├── challenge-2019-validation-metas.pkl     # get from script
   │   │   ├── OpenImages
   │   │   │   ├── train           # training images
   │   │   │   ├── test            # testing images
   │   │   │   ├── validation      # validation images
   ```
## Object365
1. You need to download and extract Objects365 dataset. Users can download Objects365 V2 by using `tools/misc/download_dataset.py`.

   **Usage**

   ```shell
   python tools/misc/download_dataset.py --dataset-name objects365v2 \
   --save-dir ${SAVING PATH} \
   --unzip \
   --delete  # Optional, delete the download zip file
   ```

   **Note:** There is no download link for Objects365 V1 right now. If you would like to download Objects365-V1, please visit [official website](http://www.objects365.org/) to concat the author.

2. The directory should be like this:

   ```none
   mmdetection
   ├── mmdet
   ├── tools
   ├── configs
   ├── data
   │   ├── Objects365
   │   │   ├── Obj365_v1
   │   │   │   ├── annotations
   │   │   │   │   ├── objects365_train.json
   │   │   │   │   ├── objects365_val.json
   │   │   │   ├── train        # training images
   │   │   │   ├── val          # validation images
   │   │   ├── Obj365_v2
   │   │   │   ├── annotations
   │   │   │   │   ├── zhiyuan_objv2_train.json
   │   │   │   │   ├── zhiyuan_objv2_val.json
   │   │   │   ├── train        # training images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   │   │   │   ├── val          # validation images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   ```
## LVIS 1.0
  ```
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
  ```
## COCO Stuff 164k

For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.

```shell
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

## VOC0712
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
## nuImages
Download samples and metadata from https://www.nuscenes.org/nuimages#download
tar -zxvf nuimages-v1.0-all-samples.tgz 

tar -zxvf nuimages-v1.0-all-metadata.tgz 
```shell
python -u tools/dataset_converters/nuimage_converter.py --data-root ${DATA_ROOT} --version ${VERSIONS} \
                                                    --out-dir ${OUT_DIR} --nproc ${NUM_WORKERS} --extra-tag ${TAG}
```

- `--data-root`: the root of the dataset, defaults to `./data/nuimages`.
- `--version`: the version of the dataset, defaults to `v1.0-mini`. To get the full dataset, please use `--version v1.0-train v1.0-val v1.0-mini`
- `--out-dir`: the output directory of annotations and semantic masks, defaults to `./data/nuimages/annotations/`.
- `--nproc`: number of workers for data preparation, defaults to `4`. Larger number could reduce the preparation time as images are processed in parallel.
- `--extra-tag`: extra tag of the annotations, defaults to `nuimages`. This can be used to separate different annotations processed in different time for study.
```shell
python -u tools/dataset_converters/nuimage_converter.py --data-root ./data/nuimages --version v1.0-train v1.0-val v1.0-mini \
                                                    --out-dir ./data/nuimages/annotations/ --nproc 8 
```

## nuImages Segmentation
python tools/dataset_preprocess/link_nuimage_seg.py

## OpenImages Instance Segmentation
```
cd data/OpenImages
mkdir segmentation
cd segmentation 
mkdir train
mkdir validation
mkdir test
python tools/dataset_preprocess/download_open_inseg.py
```
sort instances by ImageID:
```
python tools/dataset_preprocess/sort_instances.py
```
this command will create `train-annotations-object-segmentation_sort_resize.csv` and `validation-annotations-object-segmentation_sort_resize.csv` in `data/OpenImages/annotations`.

## Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/dataset_converters/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

## BDD100K

- You could download BDD100k datasets from  [here](https://bdd-data.berkeley.edu/) after  registration.
- Or download from OpenDataLab

- You can download images and masks by clicking  `10K Images` button and `Segmentation` button.

- After download, unzip by the following instructions:

  ```bash
  unzip ~/bdd100k_images_10k.zip -d ~/mmsegmentation/data/
  unzip ~/bdd100k_sem_seg_labels_trainval.zip -d ~/mmsegmentation/data/
  ```

- And get
```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json

## Mapillary Vistas Datasets

- The dataset could be download [here](https://www.mapillary.com/dataset/vistas) after registration.

- Mapillary Vistas Dataset use 8-bit with color-palette to store labels. No conversion operation is required.

- Assumption you have put the dataset zip file in `mmsegmentation/data/mapillary`

- Please run the following commands to unzip dataset.

  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```

- After unzip, you will get Mapillary Vistas Dataset like this structure. Semantic segmentation mask labels in `labels` folder.

## CCS(CC3M+CC12M+SBU)
```
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered.json
```
remove failed images:
```
python tools/dataset_preprocess/mask_ccs.py
```
# Visual Genome
```
openxlab dataset get --dataset-repo OpenDataLab/Visual_Genome_Dataset_V1_dot_2
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_caption.json
python tools/dataset_preprocess/vg_caption.py
```

# refcoco, refcoco+,refcocog
```
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
```
- use images from coco 2014

- use refs(unc).p for refcoco+ and refcoco

- use refs(umd).p for refcocog

# refclef
```
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
```
download image data from https://www.imageclef.org/SIAPRdata

# Flickr30K
reference: https://github.com/bianhao123/TransVG
- download images from http://hockenmaier.cs.illinois.edu/DenotationGraph/
- download anno from https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing

# Zero-Shot Dataset


## Cityscapes Segmentation

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
We provided a [script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/dataset_converters/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/cityscapes_seg.py data/cityscapes --nproc 8
```

## Cityscapes Detection and Instance Segmentation

The cityscapes annotations have to be converted into the coco format using `tools/dataset_converters/cityscapes.py`:

```shell
pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py ./data/cityscapes --nproc 8 --out-dir ./data/cityscapes/annotations
```

## KITTI
```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
```
download train.txt, val.txt, test.txt from  https://github.com/open-mmlab/OpenPCDet/blob/master/data/kitti/ImageSets/train.txt
convert kitti to coco format:
```
python tools/dataset_preprocess/kitti2coco.py
```
reference: https://github.com/mingkyun/kitti_mmdetection

## SUN RGB-D
reference: https://github.com/Barchid/RGBD-Seg

download sunrgbd:
```
python tools/dataset_preprocess/prepare_sunrgbd.py
```
convert sunrgbd to mmsegmentation format:
```
python tools/dataset_preprocess/organize_sunrgbd.py
```



## NoCaps
```
mkdir nocaps && cd nocaps
ln -s /u/sshi/workspace_ptmp/workspace_hy/dataset/OpenImages/OpenImages/validation ./images
mkdir annotations && annotations
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json
```

## CrowdHuman
download data from https://www.crowdhuman.org/

convert to coco:
```
python ./tools/dataset_converters/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations
```

# Few-Shot Dataset
## DRIVE
The training and validation set of DRIVE could be download from [kaggle](https://www.kaggle.com/datasets/linjianhua/drivedataset/data?select=DRIVE).

To convert DRIVE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/drive.py /path/to/DRIVE
```

The script will make directory structure automatically.

## LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/loveda.py /path/to/loveDA
```

Using trained model to predict test set of LoveDA and submit it to server can be found [here](https://codalab.lisn.upsaclay.fr/competitions/421).

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).
## Potsdam
The [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx).
Or download on [BaiduNetdisk](https://pan.baidu.com/s/1K-cLVZnd1X7d8c26FQ-nGg?pwd=mseg)，password：mseg, [Google Drive](https://drive.google.com/drive/folders/1w3EJuyUGet6_qmLwGAWZ9vw5ogeG0zLz?usp=sharing) and [OpenDataLab](https://opendatalab.com/ISPRS_Potsdam/download).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required. Create a folder containing these two zip files.

For Potsdam dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 3456 images for training and 2016 images for validation.
## WIDERFace
To use the WIDER Face dataset you need to download it
and extract to the `data/WIDERFace` folder. Annotation in the VOC format
can be found in this [repo](https://github.com/sovrasov/wider-face-pascal-voc-annotations.git).
You should move the annotation files from `WIDER_train_annotations` and `WIDER_val_annotations` folders
to the `Annotation` folders inside the corresponding directories `WIDER_train` and `WIDER_val`.
Also annotation lists `val.txt` and `train.txt` should be copied to `data/WIDERFace` from `WIDER_train_annotations` and `WIDER_val_annotations`.
## DeepFashion
[MMFashion](https://github.com/open-mmlab/mmfashion) develops "fashion parsing and segmentation" module
based on the dataset
[DeepFashion-Inshop](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?usp=sharing).
Its annotation follows COCO style.
To use it, you need to first download the data. Note that we only use "img_highres" in this task.
The file tree should be like this:

```sh
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── DeepFashion
│   │   ├── In-shop
|   │   │   ├── Anno
|   │   │   │   ├── segmentation
|   │   │   │   |   ├── DeepFashion_segmentation_train.json
|   │   │   │   |   ├── DeepFashion_segmentation_query.json
|   │   │   │   |   ├── DeepFashion_segmentation_gallery.json
|   │   │   │   ├── list_bbox_inshop.txt
|   │   │   │   ├── list_description_inshop.json
|   │   │   │   ├── list_item_inshop.txt
|   │   │   │   └── list_landmarks_inshop.txt
|   │   │   ├── Eval
|   │   │   │   └── list_eval_partition.txt
|   │   │   ├── Img
|   │   │   │   ├── img
|   │   │   │   │   ├──XXX.jpg
|   │   │   │   ├── img_highres
|   │   │   │   └── ├──XXX.jpg

```