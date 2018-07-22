#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : training
"""

import argparse
import opts
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_soccer_dataset
from utils.utils import get_logger
from onmt.decoders.decoder import StdRNNDecoder
from onmt.encoders.encoder import RNNEncoder
from onmt.models.model import NMTModel
from torch import nn


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    #  torch.manual_seed(opt.seed)

    return opt


def train(opt, logger=None):
    "training process"

    # Create dataset iterator
    data_dir = "./data_syn"
    train_iter, test_iter, val_iter = create_soccer_dataset(
        train_dir=f"./{data_dir}/train",
        test_dir=f"./{data_dir}/test",
        valid_dir=f"./{data_dir}/val"
    )
    device = torch.device(opt.device)

    encoder = RNNEncoder(
        rnn_type="GRU", bidirectional=False,
        num_layers=1, vocab_size=10,
        word_dim=5, hidden_size=10
    ).to(device)

    decoder_emb = nn.Embedding(100, 20)
    decoder = StdRNNDecoder(
        rnn_type="GRU",
        bidirectional_encoder=False,
        num_layers=1,
        hidden_size=30,
        embeddings=decoder_emb
    ).to(device)

    model = NMTModel(
        encoder=encoder,
        decoder=decoder,
        multigpu=False
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))

    # Start training
    for epoch in range(1, int(opt.epoch) + 1):
        start_time = time.time()
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()

            src, src_lengths = batch.src[0], batch.src[1]
            tgt, tgt_lengths = batch.tgt[0], batch.tgt[1]

            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)
            tgt_lengths = tgt_lengths.to(device)
            decoder_outputs, attns, dec_state = \
                model(src, tgt, src_lengths)


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("It's a test")
    train(opt, logger)
