# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import lmdb
import os

from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config


""" Utils Functions """

import random

import numpy as np
import torch
import sys


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                  , training_rate=training_rate, vali_rate=0.1
                                                                                                  , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_pretrain_dataset_gw(data, training_rate, seed=None):
    set_seeds(seed)
    data_train, data_vali = partition_and_reshape_pretrain_gw(data, training_rate=training_rate, vali_rate=0.1, change_shape=False)
    return data_train, data_vali


def prepare_pretrain_dataset_gw_lmdb(dataset_cfg, training_rate):
    # train_size = int(dataset_cfg.size * training_rate)
    # test_start = train_size
    # test_size = dataset_cfg.size - train_size
    # return 0, train_size, test_start, test_size
    train_size = int(dataset_cfg.size * training_rate)
    test_size = dataset_cfg.size - train_size
    train_start = test_size
    return train_start, train_size, 0, test_size


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1
                                , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test



def prepare_classifier_dataset_gw(args, data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                                  , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    # data_train, label_train, data_vali, label_vali, data_test, label_test \
    #     = partition_and_reshape_classifier_gw(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1, change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape_classifier_wwc(args, data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1, change_shape=change_shape, merge=merge, merge_mode=merge_mode)

    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test

def prepare_classifier_dataset_lzj( data, labels ,label_test_file_name_save_path, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                                    , merge=0, merge_mode='all', seed=None, balance=False ,):

    set_seeds(seed)
    # data_train, label_train, data_vali, label_vali, data_test, label_test \
    #     = partition_and_reshape_classifier_gw(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1, change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape_classifier_lzj(data, labels ,label_test_file_name_save_path, label_index=label_index, training_rate=training_rate, vali_rate=0.0, change_shape=change_shape, merge=merge, merge_mode=merge_mode)

    set_seeds(seed)
    # if balance:
    #     data_train_label, label_train_label, _, _ \
    #         = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    # else:
    #     data_train_label, label_train_label, _, _ \
    #         = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train ,label_train ,data_vali ,label_vali, data_test, label_test
def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num +vali_num, ...]
    data_test = data[train_num +vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num +vali_num, ..., label_index] - t
    label_test = labels[train_num +vali_num:, ..., label_index] - t
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
    print('Train Size: %d, Vali Size: %d, Test Size: %d' %
    (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def partition_and_reshape_pretrain_gw(data, training_rate=0.9, vali_rate=0.1, change_shape=True, merge=0,
                                      merge_mode='all', shuffle=True):
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


def partition_and_reshape_classifier_lzj(data, labels, label_test_file_name_save_path, label_index=0, training_rate=0.8,
                                         vali_rate=0.1, change_shape=True, merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]

    # 提取所有唯一用户ID（假设labels的第一列为用户ID）
    user_ids = np.unique(labels[:, :, 0])
    # np.random.shuffle(user_ids)  # 随机打乱用户顺序.  20250523 lzj 关闭

    # 按比例划分用户
    total_users = len(user_ids)
    train_users_cnt = int(total_users * training_rate)

    val_users_cnt = int(total_users * vali_rate)
    test_users_cnt = total_users - train_users_cnt - val_users_cnt

    # 至少保留1个用户防止空切片
    train_users_cnt = max(0, train_users_cnt)
    val_users_cnt = max(0, val_users_cnt)
    test_users_cnt = max(1, test_users_cnt)

    train_users = user_ids[:train_users_cnt]
    val_users = user_ids[train_users_cnt:train_users_cnt + val_users_cnt]
    test_users = user_ids[train_users_cnt + val_users_cnt:]
    # print(f"lzj test: {test_users}")

    print(f"Total Users: {total_users}")
    print(f"Train Users: {len(train_users)}, Val Users: {len(val_users)}, Test Users: {len(test_users)}")

    # 创建用户掩码
    train_mask = np.isin(labels[:, :, 0], train_users)
    val_mask = np.isin(labels[:, :, 0], val_users)
    test_mask = np.isin(labels[:, :, 0], test_users)

    # 分割数据和标签
    data_train = data[train_mask[:, 0]]
    data_vali = data[val_mask[:, 0]]
    data_test = data[test_mask[:, 0]]

    label_train = labels[train_mask[:, 0], ..., label_index]
    label_vali = labels[val_mask[:, 0], ..., label_index]
    label_test = labels[test_mask[:, 0], ..., label_index]
    print(f" the label_test size: {labels[test_mask[:, 0]].shape}")

    # #  保存下测试样本和对应的label
    # file_name_save = os.path.join(args.dataset_path, args.dataset + '/label_testset.npy')
    # np.save(file_name_save,labels[test_mask[:,0],...,:])

    # file_name_save = os.path.join(args.dataset_path, args.dataset + '/data_testset.npy')
    # np.save(file_name_save,data_test)
    file_name_save1 = os.path.join(label_test_file_name_save_path, 'label_test_for_lzj.npy')
    label_res_save = labels[test_mask[:, 0]]
    np.save(file_name_save1, label_res_save[:, 0, :])

    file_name_save2 = os.path.join(label_test_file_name_save_path, 'data_test_for_lzj.npy')
    np.save(file_name_save2, data_test)

    # 保留原有change_shape和merge逻辑
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)

    # 打印最终数据规模和用户信息
    print(f"Final Data Sizes:")
    print(f"Train: {label_train.shape[0]} samples ({len(train_users)} users)")
    print(f"Val: {label_vali.shape[0]} samples ({len(val_users)} users)")
    print(f"Test: {label_test.shape[0]} samples ({len(test_users)} users)")

    return data_train, label_train, data_vali, label_vali, data_test, label_test


def partition_and_reshape_classifier_wwc(args, data, labels, label_index=0, training_rate=0.8, vali_rate=0.1,
                                         change_shape=True, merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]

    # 提取所有唯一用户ID（假设labels的第一列为用户ID）
    user_ids = np.unique(labels[:, :, 0])
    # np.random.shuffle(user_ids)  # 随机打乱用户顺序.  20250523 lzj 关闭

    # 按比例划分用户
    total_users = len(user_ids)
    train_users_cnt = int(total_users * training_rate)

    val_users_cnt = int(total_users * vali_rate)
    test_users_cnt = total_users - train_users_cnt - val_users_cnt

    # 至少保留1个用户防止空切片
    train_users_cnt = max(1, train_users_cnt)
    val_users_cnt = max(1, val_users_cnt)
    test_users_cnt = max(1, test_users_cnt)

    train_users = user_ids[:train_users_cnt]
    val_users = user_ids[train_users_cnt:train_users_cnt + val_users_cnt]
    test_users = user_ids[train_users_cnt + val_users_cnt:]
    # print(f"lzj test: {test_users}")

    print(f"Total Users: {total_users}")
    print(f"Train Users: {len(train_users)}, Val Users: {len(val_users)}, Test Users: {len(test_users)}")

    # 创建用户掩码
    train_mask = np.isin(labels[:, :, 0], train_users)
    val_mask = np.isin(labels[:, :, 0], val_users)
    test_mask = np.isin(labels[:, :, 0], test_users)

    # 分割数据和标签
    data_train = data[train_mask[:, 0]]
    data_vali = data[val_mask[:, 0]]
    data_test = data[test_mask[:, 0]]

    label_train = labels[train_mask[:, 0], ..., label_index]
    label_vali = labels[val_mask[:, 0], ..., label_index]
    label_test = labels[test_mask[:, 0], ..., label_index]

    #  保存下测试样本和对应的label
    file_name_save = os.path.join(args.dataset_path, args.dataset + '/label_testset.npy')
    np.save(file_name_save, labels[test_mask[:, 0], ..., :])

    file_name_save = os.path.join(args.dataset_path, args.dataset + '/data_testset.npy')
    np.save(file_name_save, data_test)

    # 保留原有change_shape和merge逻辑
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)

    # file_name_save1 = "../data/dataset/knights_stair_recognition_trainset_zjs/label_only_testset.npy"
    # np.save(file_name_save1,label_test)
    # 打印最终数据规模和用户信息
    print(f"Final Data Sizes:")
    print(f"Train: {label_train.shape[0]} samples ({len(train_users)} users)")
    print(f"Val: {label_vali.shape[0]} samples ({len(val_users)} users)")
    print(f"Test: {label_test.shape[0]} samples ({len(test_users)} users)")

    return data_train, label_train, data_vali, label_vali, data_test, label_test


def partition_and_reshape_classifier_gw(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1,
                                        change_shape=True, merge=0, merge_mode='all', shuffle=True):
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
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (
    label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)

    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))

    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print(
        'Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                                              ,
                                                                              label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.mag_norm = 10.0
        self.gamma = gamma
        # print('完成初始化 utils 中数据归一化方法')

    def __call__(self, instance):
        # return instance
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 3 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        # if instance_new.shape[1] == 9 and self.norm_mag:
        #     mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
        #     mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
        #     instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        if instance_new.shape[1] == 9 and self.norm_mag:
            instance_new[:, 6:9] = instance_new[:, 6:9] * self.mag_norm
        return instance_new




class Preprocess4Rotation(Pipeline):
    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen
        print('完成初始化 utils 中旋转数据增强方法')

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        # return instance 
        # print('调用随机旋转方法')
        # print(instance)
        instance_new = instance.reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = special_ortho_group.rvs(self.sensor_dimen)
        for i in range(instance_new.shape[1]):
            instance_new[:, i, :] = np.dot(instance_new[:, i, :], rotation_matrix)
        re = instance_new.reshape(instance.shape[0], instance.shape[1])
        # print(re)
        return re


class Preprocess4Scale(Pipeline):
    def __init__(self, scale=0.4, scale_range=[0.95, 1.05]):
        super().__init__()
        self.scale = scale
        self.scale_range = scale_range

    def __call__(self, instance):
        factors = np.random.random(instance.shape[1])
        factors = factors * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        return instance * factors


class Preprocess4Sample(Pipeline):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def __call__(self, instance):
        if instance.shape[0] == self.seq_len:
            return instance
        index_rand = np.random.randint(0, high=instance.shape[0] - self.seq_len)
        return instance[index_rand:index_rand + self.seq_len, :]


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram, goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)


class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


class BalancedMultiAugmentor:
    """多分类平衡增强器（自动平衡所有类别到最大类数量）"""

    def __init__(self, augmentations, copy_original=True):
        self.augmentations = augmentations
        self.copy_original = copy_original

    def apply(self, data, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)

        all_augmented_data = []
        all_augmented_labels = []

        for label, count in zip(unique_labels, counts):
            mask = (labels == label)
            class_data = data[mask]
            needed = max_count - (count if self.copy_original else 0)
            if needed <= 0:
                if self.copy_original:
                    all_augmented_data.append(class_data)
                    all_augmented_labels.append(np.full(count, label))
                continue

            augmented_data = []
            augmented_labels = []

            augment_rounds = (needed + count - 1) // count
            for _ in range(augment_rounds):
                aug = np.random.choice(self.augmentations)
                indices = np.random.permutation(len(class_data))
                augmented = [aug(class_data[i]) for i in indices[:needed]]
                augmented_data.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
                if len(augmented_data) >= needed:
                    break

            augmented_data = augmented_data[:needed]
            augmented_labels = augmented_labels[:needed]

            if self.copy_original:
                final_data = np.concatenate([class_data, augmented_data], axis=0)
                final_labels = np.concatenate([np.full(count, label), augmented_labels], axis=0)
            else:
                final_data = np.array(augmented_data)
                final_labels = np.array(augmented_labels)

            all_augmented_data.append(final_data)
            all_augmented_labels.append(final_labels)

        balanced_data = np.concatenate(all_augmented_data, axis=0)
        balanced_labels = np.concatenate(all_augmented_labels, axis=0)
        # 统计平衡后的标签分布
        unique_labels_after, counts_after = np.unique(balanced_labels, return_counts=True)
        print("王文超新增的数据平衡器平衡后的样本分布如下:")
        for label, count in zip(unique_labels_after, counts_after):
            print(f"Label {label}: {count} samples")
        return balanced_data, balanced_labels


class Preprocess4Noise(Preprocess4Normalization):
    """ 添加高斯噪声增强 """

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, instance):
        noise = np.random.normal(scale=self.noise_level, size=instance.shape)
        return instance + noise


class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)


class LIBERTDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return len(self.data)


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
            self.env = lmdb.open(dataset_cfg.root, max_readers=126, readonly=True, lock=False, readahead=False,
                                 meminit=False)
        else:
            self.env = lmdb.open(dataset_cfg.root, max_readers=126, readonly=True, lock=False, readahead=False,
                                 meminit=False)

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
    parser.add_argument('dataset', type=str, nargs='?', default='eleme', help='Dataset name',
                        choices=['hhar', 'motion', 'uci', 'shoaib', 'eleme_liyang', 'eleme_liyang_augmentation',
                                 'train_set_0522_from_wwc', 'eleme_pretrain', 'eleme_yangzhou20',
                                 'eleme_yangzhou20_upsample', 'eleme_yangzhou20_10class', 'eleme_yangzhou20_3class',
                                 'eleme_yangzhou20_7class', 'eleme_press_2class', 'wwc_press_2class',
                                 'wwc_press_pretrain', 'eleme_yangzhou20_3class_input3', '(ceshi)trainset_zjs_dim9',
                                 '(ceshi)trainset_zjs_dim6', 'knights_stair_recognition_trainset_zjs',
                                 'knights_stair_recognition_trainset_zjs_window500',
                                 'knights_stair_recognition_trainset_filtered_zjs_window500',
                                 'knights_stair_recognition_trainset_deep_filtered_zjs_window500',
                                 'knights_stair_recognition_trainset_zjs_window500_v2',
                                 'knights_stair_recognition_trainset_zjs_window500_4class_v1',
                                 'knights_stair_recognition_trainset_zjs_window500_4class_input6_v2',
                                 'knights_stair_recognition_trainset_zjs_window100_4class_input6',
                                 'knights_HAR_motional_trainset_zjs_window100_input6',
                                 'knights_FUSION_HAR_trainset_zjs_window100_input6',
                                 'knights_FUSION_HAR_static_trainset_zjs_window100_input6'])
    parser.add_argument('dataset_version', type=str, nargs='?', default='10_20', help='Dataset version',
                        choices=['10_100', '20_120', '10_20', '10_150', '20_200', 'zhiti_20_200', '20_500', '20_100'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train,
                        help='Training config json file path')
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
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'merge'])
    parser.add_argument('dataset_version', type=str, help='Dataset version', choices=['10_100', '20_120'])
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
    print('\nmodel_cfg', model_cfg)
    print('\ntrain_cfg', train_cfg)
    print('\nmask_cfg', mask_cfg)
    print('\ndataset_cfg', dataset_cfg)
    print('\ntrain data path', args.data_path)
    print('\nlabel data path', args.label_path)
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)

    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    # print(data)
    # print(labels)
    print('\ndata.shape', data.shape)
    print('\nlabels.shape', labels.shape)
    print('pretrain_data_load_success', dataset_cfg)
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


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    print('\n model_bert_cfg', model_bert_cfg)
    print('\n model_classifier_cfg', model_classifier_cfg)
    print('\n train_cfg', train_cfg)
    print('\n dataset_cfg', dataset_cfg)

    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    print('\n train data path', args.data_path)
    print('\n label data path', args.label_path)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    print('\ndata.shape', data.shape)
    print('\nlabels.shape', labels.shape)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)