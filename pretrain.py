#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device, LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, \
    prepare_classifier_dataset, prepare_pretrain_dataset, Preprocess4Normalization, Preprocess4Mask, Preprocess4Sample
from utils import load_pretrain_data_config_gw, prepare_pretrain_dataset_gw
# from utils import set_seeds, get_device, LMDBDataset4Pretrain, handle_argv, load_pretrain_data_config_gw_lmdb, prepare_classifier_dataset, prepare_pretrain_dataset_gw_lmdb, Preprocess4Normalization, Preprocess4Rotation, Preprocess4Mask


def main(args, training_rate):
    # data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    data, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config_gw(args)
    # train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config_gw_lmdb(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Sample(model_cfg.seq_len), Preprocess4Mask(mask_cfg)]
    # pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Rotation(), Preprocess4Mask(mask_cfg)]

    # data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    data_train, data_test = prepare_pretrain_dataset_gw(data, training_rate, seed=train_cfg.seed)
    # train_start, train_size, test_start, test_size = prepare_pretrain_dataset_gw_lmdb(dataset_cfg, training_rate)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    # data_set_train = LMDBDataset4Pretrain(dataset_cfg, train_start, train_size, True, pipeline=pipeline)
    # data_set_test = LMDBDataset4Pretrain(dataset_cfg, test_start, test_size, False, pipeline=pipeline)

    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, num_workers=8)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size, num_workers=4)
    # data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=train_cfg.batch_size, pin_memory=True, num_workers=256)
    # data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size, pin_memory=True, num_workers=256)

    model = LIMUBertModel4Pretrain(model_cfg)
    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch #
        seq_recon = model(mask_seqs, masked_pos) #
        loss_lm = criterion(seq_recon, seqs) # for masked LM
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=args.pretrain_model, data_parallel=True)
        # trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=args.pretrain_model, data_parallel=False)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)
        # trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None, data_parallel=False)


if __name__ == "__main__":
    import sys
    # 设置默认参数
    # if len(sys.argv) < 2:
        # sys.argv.extend(['v4', 'eleme_yangzhou20_3class','20_200','-s hhar6_pretrain'])
        # sys.argv.extend(['wwc', 'wwc_press_pretrain','20_200','-fvhar_pretrain_9','-svhar_pretrain_9'])
        # sys.argv.extend(['wwc', 'train_set_0522_from_wwc','20_200','-svhar_pretrain_66'])
        # sys.argv.extend(['zjs', '(ceshi)trainset_zjs_dim6','20_200','-s vhar_pretrain_zjs_1'])
        # sys.argv.extend(['zjs', 'knights_HAR_motional_trainset_zjs_window100_input6','20_100','-svhar_pretrain_zjs_v3'])
        # sys.argv.extend(['wwc', 'wwc_press_2class','20_200','-fvhar_pretrain_9','-svhar_pretrain_9'])

    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.9
    # training_rate = 0.99
    print(args)
    main(args, training_rate)