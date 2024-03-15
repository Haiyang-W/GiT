# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import argparse as ap
import urllib.request
from zipfile import ZipFile

import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

# see: http://rgbd.cs.princeton.edu/ in section Data and Annotation
DATASET_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
DATASET_TOOLBOX_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)


if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare SUNRGBD dataset for segmentation.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    args = parser.parse_args()

    # expand user
    output_path = os.path.expanduser(args.output_path)

    os.makedirs(output_path, exist_ok=True)

    toolbox_dir = os.path.join(output_path, 'SUNRGBDtoolbox')

    # download and extract data
    if not os.path.exists(toolbox_dir):
        zip_file_path = os.path.join(output_path, 'SUNRGBDtoolbox.zip')
        download_file(DATASET_TOOLBOX_URL, zip_file_path,
                      display_progressbar=True)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_file_path))
        os.remove(zip_file_path)

    zip_file_path = os.path.join(output_path, 'SUNRGBD.zip')
    if not os.path.exists(zip_file_path):
        download_file(DATASET_URL, zip_file_path,
                      display_progressbar=True)
    print('Extract images')
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_file_path))
    os.remove(zip_file_path)

    # extract labels from SUNRGBD toolbox
    print('Extract labels from SUNRGBD toolbox')
    SUNRGBDMeta_dir = os.path.join(toolbox_dir, 'Metadata/SUNRGBDMeta.mat')
    allsplit_dir = os.path.join(toolbox_dir, 'traintestSUNRGBD/allsplit.mat')
    SUNRGBD2Dseg_dir = os.path.join(toolbox_dir, 'Metadata/SUNRGBD2Dseg.mat')
    img_dir_train = []
    depth_dir_train = []
    label_dir_train = []
    img_dir_test = []
    depth_dir_test = []
    label_dir_test = []

    SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

    # load the data from the matlab file
    SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                   struct_as_record=False)['SUNRGBDMeta']
    split = scipy.io.loadmat(allsplit_dir, squeeze_me=True,
                             struct_as_record=False)
    split_train = split['alltrain']

    seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

    for i, meta in tqdm(enumerate(SUNRGBDMeta)):
        meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
        real_dir = meta_dir.split('/n/fs/sun3d/data/SUNRGBD/')[1]
        depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
        rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

        label_path = os.path.join(real_dir, 'label/label.npy')
        label_path_full = os.path.join(output_path, 'SUNRGBD', label_path)

        # save segmentation (label_path) as numpy array
        if not os.path.exists(label_path_full):
            os.makedirs(os.path.dirname(label_path_full), exist_ok=True)
            label = np.array(
                SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0)).\
                astype(np.uint8)
            np.save(label_path_full, label)

        if meta_dir in split_train:
            img_dir_train.append(os.path.join('SUNRGBD', rgb_path))
            depth_dir_train.append(os.path.join('SUNRGBD', depth_bfx_path))
            label_dir_train.append(os.path.join('SUNRGBD', label_path))
        else:
            img_dir_test.append(os.path.join('SUNRGBD', rgb_path))
            depth_dir_test.append(os.path.join('SUNRGBD', depth_bfx_path))
            label_dir_test.append(os.path.join('SUNRGBD', label_path))

    # write file lists
    def _write_list_to_file(list_, filepath):
        with open(os.path.join(output_path, filepath), 'w') as f:
            f.write('\n'.join(list_))
        print('written file {}'.format(filepath))

    _write_list_to_file(img_dir_train, 'train_rgb.txt')
    _write_list_to_file(depth_dir_train, 'train_depth.txt')
    _write_list_to_file(label_dir_train, 'train_label.txt')
    _write_list_to_file(img_dir_test, 'test_rgb.txt')
    _write_list_to_file(depth_dir_test, 'test_depth.txt')
    _write_list_to_file(label_dir_test, 'test_label.txt')
