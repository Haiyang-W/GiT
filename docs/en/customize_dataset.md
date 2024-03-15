## Test zero-shot and few-shot performance on customize dataset
### Zero-shot
Take cityscapes for example, we test zero-shot performance of GiT.
1. **Write dataset class.** The dataset class format should align with mmdetection standards. We recommend inheriting from existing classes, e.g., CityscapesDataset from CocoDataset. Key modification is needed in the METAINFO 'classes' section, ensuring class order corresponds to annotated IDs. GiT relies on this alignment for accurate classification based on class names.
```
@DATASETS.register_module()
class CityscapesDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'),
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    }
```
2. **Prepare dataset config.** GiT requires each dataset to provide parameters such as task name and class count. For different datasets with the same task, only the 'classes_num' needs to be modified.
```
classes_num = 8
cityscapes_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=classes_num,
    num_vocal=global_bin+1+classes_num+1,
    total_num_vocal=global_bin+1+classes_num+1,
    max_decoder_length=5,
    global_only_image=True)
```
3. **Prepare dataset pipeline.** In the testing pipeline, 'Resize' and 'AddMetaInfo' are mandatory steps. This is because GiT only accepts images of fixed sizes, and it's necessary to include the previous configuration in the data information.
```
cityscapes_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=classes_num,
                                                        num_vocal=classes_num+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=cityscapes_det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]
```

4. **Prepare dataloader and eval metric.** Consistent with mmdetection. Please follow [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/docs/en/advanced_guides/customize_dataset.md

### Few-shot
The configuration for few-shot is essentially the same as zero-shot, but with the addition of training pipelines and the selection of sample quantity.

1. **Adding Training Pipelines.** Consistent with the testing process, it is crucial to ensure that the final images have a fixed size. Additionally, the `AddMetaInfo` step must be retained in the training pipeline.

2. **Selecting Sample Quantity.** We default to using 5-shot, where for a dataset with N classes, 5*N images are sampled. In the dataset class, we introduce the parameter `support_num` to specify the corresponding number of samples.
```
if self.support_num != -1:
    print(data_list[:self.support_num])
    return data_list[:self.support_num]
else:
    return data_list
```

In the configuration file, we only need to add the `support_num` parameter in the `train_dataloader` section:
```
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(type='DeepFashionDataset',
                data_root='data/DeepFashion/In-shop/',
                ann_file='Anno/segmentation/DeepFashion_segmentation_train.json',
                data_prefix=dict(img='Img/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                support_num=15*5,
                pipeline=deepfashion_det_train_pipeline,
                return_classes=True,
                backend_args=backend_args)
    )
```