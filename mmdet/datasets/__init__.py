# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset
from .coco_caption import COCOCaption
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .ade import ADE20KDataset
from .basesegdataset import BaseSegDataset
from .refcoco import RefCOCO
from .ccs_caption import CCSCaption
from .vg_caption import VGCaption
from .coco_stuff import COCOStuffDataset
from .nuimage import NuimageDataset
from .openimages_inseg import OpenImagesDatasetInseg
from .nuimage_seg import NuimageSegDataset
from .bdd100k import BDD100KDataset
from .mapillary import MapillaryDataset_v2
from .pascal_context import PascalContextDataset,PascalContextDataset59
from .flickr30k import Flickr30K
from .cityscapes_seg import CityscapesSegDataset
from .drive import DRIVEDataset
from .bdd100k_det import BDD100KDetDataset
from .ade20k import ADE20KInstanceDataset
from .bdd100k_ins import BDD100KInsDataset
from .v3det import V3DetDataset
from .sun_rgbd import SunRGBDDataset
from .nocaps import NoCaps
from .potsdam import PotsdamDataset
from .loveda import LoveDADataset
__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset', 'ADE20KDataset', 'BaseSegDataset','COCOCaption','RefCOCO','CCSCaption','VGCaption',
    'COCOStuffDataset','NuimageDataset','OpenImagesDatasetInseg','NuimageSegDataset','BDD100KDataset','MapillaryDataset_v2','PascalContextDataset','PascalContextDataset59',
    'Flickr30K','CityscapesSegDataset','DRIVEDataset','BDD100KDetDataset','ADE20KInstanceDataset','BDD100KInsDataset','V3DetDataset','SunRGBDDataset','NoCaps',
    'PotsdamDataset','LoveDADataset',
]
