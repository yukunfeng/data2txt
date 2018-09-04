#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import argparse
import string
import torchtext
import torch
import spacy
from utils.utils import word_ids_to_sentence
import opts_json


# Using spacy to tokenize text
spacy_en = spacy.load('en')
# Add <unk> special case is due to wiki text which has raw <unk>
#  spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])


def tokenize(sequence):
    """tokenize sequence"""
    return [item.text for item in spacy_en.tokenizer(sequence)]


def filter_pred(example):
    """filter predicate"""
    src = example.src
    if len(src) == 0:
        return False
    return True


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts_json.preprocess_opts(parser)

    opt = parser.parse_args()
    #  torch.manual_seed(opt.seed)

    return opt


def define_fields():

    max_sources = 38
    res = []
    CMT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=True
    )

    LEN = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )
    res.append(("cmt", CMT))
    res.append(("len", LEN))

    EVENT_ID = torchtext.data.Field(
        sequential=True,
        use_vocab=True
    )
    TYPE_ID = torchtext.data.Field(
        sequential=True,
        use_vocab=True
    )
    MINUTE = torchtext.data.Field(
        preprocessing=lambda x: float(x),
        dtype=torch.float,
        sequential=False,
        use_vocab=False
    )
    SECOND = torchtext.data.Field(
        preprocessing=lambda x: float(x),
        dtype=torch.float,
        sequential=False,
        use_vocab=False
    )
    OUTCOME = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )
    POSITION = torchtext.data.Field(
        preprocessing=lambda x: float(x),
        dtype=torch.float,
        sequential=False,
        use_vocab=False
    )
    END_POSITION = torchtext.data.Field(
        preprocessing=lambda x: float(x),
        dtype=torch.float,
        sequential=False,
        use_vocab=False
    )
    QUANTIFIER = torchtext.data.Field(
        sequential=True,
        use_vocab=True
    )
    for i in range(max_sources):
        res.append((f"event_id_{i}", EVENT_ID))
        res.append((f"type_id_{i}", TYPE_ID))
        res.append((f"minute_{i}", MINUTE))
        res.append((f"second_{i}", SECOND))
        res.append((f"outcome_{i}", OUTCOME))
        res.append((f"position_x_{i}", POSITION))
        res.append((f"position_y_{i}", POSITION))
        res.append((f"end_position_x{i}", END_POSITION))
        res.append((f"end_position_y{i}", END_POSITION))
        res.append((f"quantifier_{i}", QUANTIFIER))
    return res


def get_field_by_name(fields, field_name):
    for name, field in fields:
        if field_name == name:
            return field



def create_soccer_dataset(opt):
    """create soccer dataset.
    :returns: iterators for train, test and valid dataset

    """
    soccer_fields = define_fields()
    # Here the returned order
    train, valid, test = torchtext.data.TabularDataset.splits(
        fields=soccer_fields,
        path="",
        train=opt.train_path,
        validation=opt.valid_path,
        test=opt.test_path,
        format="tsv"
    )
    # Each Field only uses its own column to build vocab
    for name, field in soccer_fields:
        if field.use_vocab:
            field.build_vocab(train)

    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, valid),
        batch_sizes=(opt.batch_size, opt.batch_size),
        device=opt.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.cmt),
        repeat=False
    )
    test_iter = None
    
    return (soccer_fields, train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    opt = parse_args()
    soccer_fields, train_iter, test_iter, val_iter = create_soccer_dataset(opt)
    for counter, batch in enumerate(train_iter, 1):
        src = batch.cmt
        words_src = word_ids_to_sentence(src, soccer_fields[0][1].vocab)
        print(f"{words_src}")
        for name, field in soccer_fields:
            obj = getattr(batch, name)
            if field.use_vocab:
                obj = word_ids_to_sentence(obj, field.vocab)
            print(name)
            print(obj)
            print("---")
        #  print(f"{soccer_fields[0][1].vocab.itos}")
        print("---------------")
        break
