#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import argparse
import string
from utils.utils import word_ids_to_sentence
import opts
import torchtext


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    #  torch.manual_seed(opt.seed)

    return opt


def create_soccer_dataset(opt):
    """create soccer dataset.
    :returns: iterators for train, test and valid dataset

    """
    train_dir = opt.train_dir + "/"
    test_dir = opt.test_dir + "/"
    valid_dir = opt.val_dir + "/"

    def tokenize(sequence):
        """tokenize sequence"""
        tok_list = sequence.split()
        tok_list = [x for x in tok_list if x not in string.punctuation]
        return tok_list

    def filter_pred(example):
        """filter predicate"""
        src = example.src
        if len(src) == 0:
            return False
        return True

    SRC = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=True,
        lower=True,
        include_lengths=True
    )

    TGT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=True,
        include_lengths=True,
        init_token='<s>',
        eos_token='</s>'
    )
    soccer_fields = [("src", SRC), ("tgt", TGT)]
    train, test, valid = torchtext.datasets.TranslationDataset.splits(
        exts=("src", "tgt"),
        fields=soccer_fields,
        path="",
        train=train_dir,
        validation=valid_dir,
        test=test_dir,
        filter_pred=filter_pred
    )
    # Each Field only uses its own column to build vocab
    TGT.build_vocab(train)
    SRC.build_vocab(train)

    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, valid),
        batch_sizes=(opt.batch_size, opt.batch_size),
        device=opt.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False
    )
    test_iter = None
    
    return (SRC, TGT, train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    data_dir = "./data_syn"
    opt = parse_args()
    SRC, TGT, train_iter, test_iter, val_iter = create_soccer_dataset(opt)
    exit(0)
    for counter, batch in enumerate(train_iter, 1):
        batch_src, src_lengths = batch.src[0], batch.src[1]
        batch_tgt, tgt_lengths = batch.tgt[0], batch.tgt[1]
        words_src = word_ids_to_sentence(batch_src, SRC.vocab)
        words_tgt = word_ids_to_sentence(batch_tgt, TGT.vocab)
        #  if 35793 in batch_tgt.view(-1):
            #  wrong_value_index = batch_tgt.view(-1).tolist().index(35793)
            #  wrong_value = TGT.vocab.itos[wrong_value_index]
            #  print(f"vocab size: {len(TGT.vocab.freqs.keys())}")
            #  print(f"vocab size: {len(TGT.vocab.itos)}")
            #  print(f"wrong_value_index: {wrong_value_index} wrong_value: {wrong_value}")
        print(f"{counter}-th batch size")
        print(f"words_src: {words_src}") 
        print(f"words_src real lengths: {src_lengths}")
        print("")
        print(f"words_tgt: {words_tgt}") 
        print(f"words_tgt real lengths: {tgt_lengths}")
        print(f"words_tgt: {batch_tgt}")
        print("---------------")
