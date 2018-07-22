#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import torchtext


def create_soccer_dataset(train_dir, test_dir, valid_dir):
    """create soccer dataset.
    :returns: iterators for train, test and valid dataset

    """
    train_dir = train_dir + "/"
    test_dir = test_dir + "/"
    valid_dir = valid_dir + "/"

    def tokenize(events):
        """tokenize events"""
        return events.split()

    EVENTS = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=True,
        lower=True
    )

    COMMENT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=True
    )
    soccer_fields = [("events", EVENTS), ("comment", COMMENT)]
    train, test, valid = torchtext.datasets.TranslationDataset.splits(
        exts=("src", "tgt"),
        fields=soccer_fields,
        path="",
        train=train_dir,
        validation=valid_dir,
        test=test_dir
    )

    # Each Field only uses its own column to build vocab
    COMMENT.build_vocab(train)
    EVENTS.build_vocab(train)
    words = EVENTS.vocab.freqs.keys()
    #  print(len(words))

    #  train_iter, val_iter = torchtext.data.Iterator.splits(
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, valid),
        batch_sizes=(2, 2),
        device="cpu",
        sort_within_batch=False,
        sort_key=lambda x: len(x.comment),
        repeat=False
    )
    test_iter = None
    
    return (train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    data_dir = "./data_syn"
    train_iter, test_iter, val_iter = create_soccer_dataset(
        train_dir=f"./{data_dir}/train",
        test_dir=f"./{data_dir}/test",
        valid_dir=f"./{data_dir}/val"
    )

    for train_batch in train_iter:
        print(train_batch.comment)
