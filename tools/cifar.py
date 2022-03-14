import os
import os.path
import pickle
import copy
import numpy as np
import torch.distributed as dist
import mmcv
from mmcv.runner import get_dist_info

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.utils import check_integrity, download_and_extract_archive


def x_u_split(num_labeled, labels, replace=False):
    class_num = len(set(labels))
    label_per_class, surplus = divmod(num_labeled, class_num)
    labels = np.array(labels)
    labeled_idx = []
    for i in range(class_num):
        idx = np.where(labels == i)[0]
        surplus_i = 1 if surplus > 0 else 0
        surplus -= surplus_i
        idx = np.random.choice(idx, label_per_class + surplus_i, replace)
        labeled_idx.extend(idx)
    unlabeled_idx = np.array(list(set(range(len(labels))) - set(labeled_idx)))
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    return labeled_idx, unlabeled_idx



def get_npy(data_path, is_train, corruption, severity, shuffle=False):
    # np.load('data/CIFAR-10-C/train/%s_%d_images.npy' % (corruption, level))

    npy_list = []
    load_list = []
    for c in corruption:
        for s in severity:
            load_list.append((c, s))
    load_list = np.array(load_list)

    if shuffle:
        order = np.random.permutation(len(corruption) * len(severity))
        load_list = load_list[order]
        print('Shuffling:', load_list)

    for i in load_list:
        c, s = i[0], int(i[1])
        assert s in [1, 2, 3, 4, 5]
        assert c in [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur',
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        if is_train:
            len_npy = 50000  # num of training images
            npy_all = np.load(data_path + '/train/%s.npy' % c)
        else:
            len_npy = 10000  # num of testing images
            npy_all = np.load(data_path + '/val/%s.npy' % c)
        npy_list.append(npy_all[(s - 1) * len_npy: s * len_npy])

    if len(npy_list) > 1:
        return np.concatenate(tuple(npy_list), axis=0)
    else:
        return npy_list[0]


def get_npy_cbar(data_path, is_train, corruption, severity, shuffle=False):
    npy_list = []
    load_list = []
    for c in corruption:
        for s in severity:
            load_list.append((c, s))
    load_list = np.array(load_list)

    if shuffle:
        order = np.random.permutation(len(corruption) * len(severity))
        load_list = load_list[order]
        print('Shuffling:', load_list)

    for i in load_list:
        c, s = i[0], int(i[1])
        assert s in [1, 2, 3, 4, 5]
        assert c in [
            "blue_noise_sample", "checkerboard_cutout", 
            "inverse_sparkles", "pinch_and_twirl",
            "ripple", "brownish_noise",
            "circular_motion_blur", "lines", 
            "sparkles", "transverse_chromatic_abberation"
        ]
        if is_train:
            len_npy = 50000  # num of training images
            npy_all = np.load(data_path + '/%s.npy' % c)
        else:
            len_npy = 10000  # num of testing images
            npy_all = np.load(data_path + '/%s.npy' % c)
        npy_list.append(npy_all[(s - 1) * len_npy: s * len_npy])

    if len(npy_list) > 1:
        return np.concatenate(tuple(npy_list), axis=0)
    else:
        return npy_list[0]


def get_npy_25(data_path, is_train, corruption, severity, shuffle=False):
    npy_list = []
    load_list = []
    for c in corruption:
        for s in severity:
            load_list.append((c, s))
    load_list = np.array(load_list)

    if shuffle:
        order = np.random.permutation(len(corruption) * len(severity))
        load_list = load_list[order]
        print('Shuffling:', load_list)

    for i in load_list:
        c, s = i[0], int(i[1])
        assert s in [1, 2, 3, 4, 5]
        assert c in [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur',
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            "blue_noise_sample", "checkerboard_cutout", 
            "inverse_sparkles", "pinch_and_twirl",
            "ripple", "brownish_noise",
            "circular_motion_blur", "lines", 
            "sparkles", "transverse_chromatic_abberation"
        ]
        if is_train:
            len_npy = 50000  # num of training images
            npy_all = np.load(data_path + '/%s.npy' % c)
        else:
            len_npy = 10000  # num of testing images
            npy_all = np.load(data_path + '/%s.npy' % c)
        npy_list.append(npy_all[(s - 1) * len_npy: s * len_npy])

    if len(npy_list) > 1:
        return np.concatenate(tuple(npy_list), axis=0)
    else:
        return npy_list[0]


@DATASETS.register_module()
class CIFAR10C(CIFAR10):
    """`CIFAR10-C <https://github.com/hendrycks/robustness>`_ Dataset.
    """

    npy_folder = 'cifar10c'

    def __init__(self, corruption, severity, shuffle=False, **kwargs):
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        self.shuffle = shuffle
        super().__init__(**kwargs)

    def load_annotations(self):

        if not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # self.imgs = []
        # self.gt_labels = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                # self.imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        # self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        # self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.imgs = get_npy(
            os.path.join(self.data_prefix, self.npy_folder),
            not self.test_mode,
            self.corruption,
            self.severity,
            self.shuffle
        )
        # CORRUPTED DATA is already N x 32 x 32 x 3
        len_c, len_s, len_l = len(self.corruption), len(self.severity), len(gt_labels)
        self.gt_labels = np.array(gt_labels * len_c * len_s)
        print('CIFAR10C', '(test)' if self.test_mode else '(train)', self.corruption, self.severity, self.imgs.shape, len(self.gt_labels))
        self._load_meta()

        # shuffle data of each single npy
        if self.shuffle:
            order = []
            for c in range(len_c):
                for s in range(len_s):
                    start = (c * len_s + s) * len_l
                    print('Shuffling: {} to {}'.format(start, start + len_l))
                    order.append(np.random.permutation([i for i in range(start, start + len_l)]))
            order = np.concatenate(order, axis=0)
            self.imgs = self.imgs[order]
            self.gt_labels = self.gt_labels[order]

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            # info = {'img': img, 'gt_label': gt_label}
            info = {
                'img': img,
                'gt_label': gt_label,
                'ori_img': copy.deepcopy(img)
            }
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR10CBAR(CIFAR10):
    """`CIFAR10-C-BAR <https://github.com/facebookresearch/augmentation-corruption>`_ Dataset.
    """

    npy_folder = 'cifar10cbar'

    def __init__(self, corruption, severity, shuffle=False, **kwargs):
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        self.shuffle = shuffle
        super().__init__(**kwargs)

    def load_annotations(self):

        if not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # self.imgs = []
        # self.gt_labels = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                # self.imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        # self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        # self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.imgs = get_npy_cbar(
            os.path.join(self.data_prefix, self.npy_folder),
            not self.test_mode,
            self.corruption,
            self.severity,
            self.shuffle
        )
        # CORRUPTED DATA is already N x 32 x 32 x 3
        len_c, len_s, len_l = len(self.corruption), len(self.severity), len(gt_labels)
        self.gt_labels = np.array(gt_labels * len_c * len_s)
        print('CIFAR10CBAR', '(test)' if self.test_mode else '(train)', self.corruption, self.severity, self.imgs.shape, len(self.gt_labels))
        self._load_meta()

        # shuffle data of each single npy
        if self.shuffle:
            order = []
            for c in range(len_c):
                for s in range(len_s):
                    start = (c * len_s + s) * len_l
                    print('Shuffling: {} to {}'.format(start, start + len_l))
                    order.append(np.random.permutation([i for i in range(start, start + len_l)]))
            order = np.concatenate(order, axis=0)
            self.imgs = self.imgs[order]
            self.gt_labels = self.gt_labels[order]

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            # info = {'img': img, 'gt_label': gt_label}
            info = {
                'img': img,
                'gt_label': gt_label,
                'ori_img': copy.deepcopy(img)
            }
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR10C25(CIFAR10):
    """
        `CIFAR10-C <https://github.com/hendrycks/robustness>`_ Dataset.
        `CIFAR10-C-BAR <https://github.com/facebookresearch/augmentation-corruption>`_ Dataset.
    """

    npy_folder = 'cifar10c25'

    def __init__(self, corruption, severity, shuffle=False, **kwargs):
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        self.shuffle = shuffle
        super().__init__(**kwargs)

    def load_annotations(self):

        if not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # self.imgs = []
        # self.gt_labels = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                # self.imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        # self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        # self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.imgs = get_npy_25(
            os.path.join(self.data_prefix, self.npy_folder),
            not self.test_mode,
            self.corruption,
            self.severity,
            self.shuffle
        )
        # CORRUPTED DATA is already N x 32 x 32 x 3
        len_c, len_s, len_l = len(self.corruption), len(self.severity), len(gt_labels)
        self.gt_labels = np.array(gt_labels * len_c * len_s)
        print('CIFAR10C25', '(test)' if self.test_mode else '(train)', self.corruption, self.severity, self.imgs.shape, len(self.gt_labels))
        self._load_meta()

        # shuffle data of each single npy
        if self.shuffle:
            order = []
            for c in range(len_c):
                for s in range(len_s):
                    start = (c * len_s + s) * len_l
                    print('Shuffling: {} to {}'.format(start, start + len_l))
                    order.append(np.random.permutation([i for i in range(start, start + len_l)]))
            order = np.concatenate(order, axis=0)
            self.imgs = self.imgs[order]
            self.gt_labels = self.gt_labels[order]

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            # info = {'img': img, 'gt_label': gt_label}
            info = {
                'img': img,
                'gt_label': gt_label,
                'ori_img': copy.deepcopy(img)
            }
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR100C(CIFAR100):
    """`CIFAR100-C <https://github.com/hendrycks/robustness>`_ Dataset.
    """

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    npy_folder = 'cifar100c'

    def __init__(self, corruption, severity, shuffle=False, shuffle_deep=False, **kwargs):
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        self.shuffle = shuffle
        self.shuffle_deep = shuffle_deep
        super().__init__(**kwargs)

    def load_annotations(self):
        if not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                # self.imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        # self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        # self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.imgs = get_npy(
            os.path.join(self.data_prefix, self.npy_folder),
            not self.test_mode,
            self.corruption,
            self.severity,
            self.shuffle
        )
        # CORRUPTED DATA is already N x 32 x 32 x 3
        len_c, len_s, len_l = len(self.corruption), len(self.severity), len(gt_labels)
        self.gt_labels = np.array(gt_labels * len_c * len_s)
        print('CIFAR100C', '(test)' if self.test_mode else '(train)',  self.corruption, self.severity, self.imgs.shape, len(self.gt_labels))
        self._load_meta()

        # shuffle data of each single npy
        if self.shuffle:
            order = []
            for c in range(len_c):
                for s in range(len_s):
                    start = (c * len_s + s) * len_l
                    print('Shuffling: {} to {}'.format(start, start + len_l))
                    order.append(np.random.permutation([i for i in range(start, start + len_l)]))
            order = np.concatenate(order, axis=0)
            self.imgs = self.imgs[order]
            self.gt_labels = self.gt_labels[order]
        elif self.shuffle_deep:
            print('Shuffling: {} to {}'.format(0, len_c * len_s * len_l))
            order = np.random.permutation([i for i in range(len_c * len_s * len_l)])
            self.imgs = self.imgs[order]
            self.gt_labels = self.gt_labels[order]

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            # info = {'img': img, 'gt_label': gt_label}
            info = {
                'img': img,
                'gt_label': gt_label,
                'ori_img': copy.deepcopy(img)
            }
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR102(BaseDataset):
    """`CIFAR10.2 <https://github.com/modestyachts/cifar-10.2>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py  # noqa: E501
    """
    
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, soft_file=None, **kwargs):
        self.soft_file = soft_file
        super(CIFAR102, self).__init__(**kwargs)

    def load_annotations(self):

        # assert self.test_mode # test data only
        if not self.test_mode:
            npy_path = self.data_prefix + '/cifar102_train.npy'
        else:
            npy_path = self.data_prefix + '/cifar102_test.npy'
        npy_data = np.load(npy_path, allow_pickle=True).item()
        self.imgs = npy_data['images']  # Nx32x32x3
        self.gt_labels = npy_data['labels']

        if self.soft_file is not None:
            soft_labels = mmcv.load(self.soft_file)

        data_infos = []
        for idx, (img, gt_label) in enumerate(zip(self.imgs, self.gt_labels)):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            if self.soft_file is not None:
                info['gt_logit'] = np.array(soft_labels[idx])
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR101(BaseDataset):
    """`CIFAR10.1 <https://github.com/modestyachts/CIFAR-10.1>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py  # noqa: E501
    """

    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, soft_file=None, **kwargs):
        self.soft_file = soft_file
        super(CIFAR101, self).__init__(**kwargs)

    def load_annotations(self):

        # assert self.test_mode # test data only
        self.imgs = np.load(self.data_prefix + '/cifar10.1_v6_data.npy')  # 2000x32x32x3
        self.gt_labels = np.load(self.data_prefix + '/cifar10.1_v6_labels.npy')
        if self.soft_file is not None:
            soft_labels = mmcv.load(self.soft_file)

        data_infos = []
        for idx, (img, gt_label) in enumerate(zip(self.imgs, self.gt_labels)):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            if self.soft_file is not None:
                info['gt_logit'] = np.array(soft_labels[idx])
            data_infos.append(info)
        return data_infos

# axel https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_train.npy;
# axel https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_test.npy;
# axel https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy;
# axel https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy
# or wget