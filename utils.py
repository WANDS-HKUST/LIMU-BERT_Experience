import argparse

from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config


""" Utils Functions """

import random

import numpy as np
import torch
import sys


def partition_and_reshape_pretrain_gw(data, training_rate=0.9, vali_rate=0.1, change_shape=True, merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:, ...]
    # if change_shape:
    #     data_train = reshape_data(data_train, merge)
    #     data_vali = reshape_data(data_vali, merge)
    # if change_shape and merge != 0:
    #     data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
    #     data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d' % (data_train.shape[0], data_vali.shape[0]))
    return data_train, data_vali

def data_loader0():
    # 扬州打标数据集，80%训练，10%验证，10%测试
    seed = 3431
    data_path = "../data/dataset/eleme_yangzhou20/data_10_20.npy"
    label_path = "../data/dataset/eleme_yangzhou20/label_10_20.npy"
    training_rate = 0.8
    vali_rate = 0.1
    label_index = 2
    feature_num = 6
    batch_size = 128

    utils.set_seeds(seed)
    data = numpy.load(data_path).astype(numpy.float64)
    labels = numpy.load(label_path).astype(numpy.float64)
    arr = numpy.arange(data.shape[0])
    numpy.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_test = data[train_num + vali_num:, ...]
    t = numpy.min(labels[:, :, label_index])
    label_test = labels[train_num + vali_num:, ..., label_index] - t
    data_test, label_test = utils.merge_dataset(data_test, label_test)
    pipeline = [utils.Preprocess4Normalization(feature_num)]
    data_set_test = utils.IMUDataset(data_test, label_test, pipeline=pipeline)
    data_loader_test = torch.utils.data.DataLoader(data_set_test, shuffle=False, batch_size=batch_size)
    return data_loader_test


def prepare_classifier_dataset_gw(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape_classifier_gw(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1, change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test

def partition_and_reshape_classifier_gw(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True, merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    # train_num = int(data.shape[0] * training_rate)
    # vali_num = int(data.shape[0] * vali_rate)
    # data_train = data[:train_num, ...]
    # data_vali = data[train_num:train_num+vali_num, ...]
    # data_test = data[train_num+vali_num:, ...]
    # t = np.min(labels[:, :, label_index])
    # label_train = labels[:train_num, ..., label_index] - t
    # label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    # label_test = labels[train_num+vali_num:, ..., label_index] - t
    vali_test_user_index = 0
    train_user = labels[:, :, 0] != vali_test_user_index
    vali_test_user = labels[:, :, 0] == vali_test_user_index
    data_train_user = data[train_user[:, 0]]
    data_vali_test_user = data[vali_test_user[:, 0]]
    label_train_user = labels[train_user[:, 0]]
    label_vali_test_user = labels[vali_test_user[:, 0]]
    train_num = int(data_train_user.shape[0] * training_rate)
    data_train = data_train_user[:train_num, ...]
    data_vali_test = data_train_user[train_num:, ...]
    # data_vali_test = np.concatenate((data_vali_test, data_vali_test_user), axis=0)
    # vali_num = int(data_vali_test.shape[0] * 0.5)
    # data_vali = data_vali_test[:vali_num, ...]
    # data_test = data_vali_test[vali_num:, ...]
    vali_num = int(data_vali_test.shape[0] * 0.5)
    data_vali_test0 = data_vali_test[:vali_num, ...]
    data_vali_test1 = data_vali_test[vali_num:, ...]
    vali_num = int(data_vali_test_user.shape[0] * 0.5)
    data_vali_test_user0 = data_vali_test_user[:vali_num, ...]
    data_vali_test_user1 = data_vali_test_user[vali_num:, ...]
    data_vali = np.concatenate((data_vali_test0, data_vali_test_user0), axis=0)
    data_test = np.concatenate((data_vali_test1, data_vali_test_user1), axis=0)
    t = np.min(labels[:, :, label_index])
    label_train = label_train_user[:train_num, ..., label_index] - t
    label_vali_test = label_train_user[train_num:, ..., label_index] - t
    label_vali_test_user = label_vali_test_user[:, ..., label_index] - t
    # label_vali_test = np.concatenate((label_vali_test, label_vali_test_user), axis=0)
    # label_vali = label_vali_test[:vali_num, ...]
    # label_test = label_vali_test[vali_num:, ...]
    vali_num = int(label_vali_test.shape[0] * 0.5)
    label_vali_test0 = label_vali_test[:vali_num, ...]
    label_vali_test1 = label_vali_test[vali_num:, ...]
    vali_num = int(label_vali_test_user.shape[0] * 0.5)
    label_vali_test_user0 = label_vali_test_user[:vali_num, ...]
    label_vali_test_user1 = label_vali_test_user[vali_num:, ...]
    label_vali = np.concatenate((label_vali_test0, label_vali_test_user0), axis=0)
    label_test = np.concatenate((label_vali_test1, label_vali_test_user1), axis=0)
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test

class LMDBDataset4Pretrain(Dataset):
    def __init__(self, dataset_cfg, start, size, shuffle, pipeline=[]):
        super().__init__()
        self.seq_len = dataset_cfg.seq_len
        self.dimension = dataset_cfg.dimension
        self.start = start
        self.size = size
        self.shuffle = shuffle
        self.pipeline = pipeline
        if shuffle:
            self.env = lmdb.open(dataset_cfg.root, max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)
        else:
            self.env = lmdb.open(dataset_cfg.root, max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            if self.shuffle:
                key = np.random.randint(self.start, self.start + self.size)
            else:
                key = index + self.start
            buf = txn.get(str(key).encode('ascii'))
            instance = np.frombuffer(buf, dtype=np.float32)
            instance = instance.reshape(self.seq_len, self.dimension)
            for proc in self.pipeline:
                instance = proc(instance)
            mask_seq, masked_pos, seq = instance
            return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return self.size


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_version', type=str, help='Model config')
    parser.add_argument('dataset', type=str, nargs='?', default='eleme', help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'eleme_liyang', 'eleme_liyang_augmentation', 'eleme_pretrain', 'eleme_yangzhou20', 'eleme_yangzhou20_upsample', 'eleme_yangzhou20_10class', 'eleme_yangzhou20_3class', 'eleme_yangzhou20_7class', 'eleme_press_2class', 'eleme_yangzhou20_3class_input3'])
    parser.add_argument('dataset_version',  type=str, nargs='?', default='10_20', help='Dataset version', choices=['10_100', '20_120', '10_20', '10_150'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json', help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1, help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model', help='The saved model name')
    parser.add_argument('-d', '--dataset_path', type=str, default='../data/dataset', help='The dataset base path')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
    return args



def handle_argv_simple():
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib','merge'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120'])
    args = parser.parse_args()
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    return args


def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_pretrain_data_config_gw(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    return data, train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_pretrain_data_config_gw_lmdb(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, mask_cfg, dataset_cfg