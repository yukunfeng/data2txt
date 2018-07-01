#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import torchtext


def create_soccer_dataset(train_path, test_path, valid_path):
    """create soccer dataset.
    :returns: iterators for train, test and valid dataset

    """
    def tokenize(events):
        """tokenize events"""
        return events.split()

    EVENTS = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=False
    )

    COMMENT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        use_vocab=False
    )
    soccer_fields = [("events", EVENTS), ("comment", COMMENT)]
    train, test, valid = torchtext.data.TabularDataset.splits(
        path="",
        train=train_path,
        validation=valid_path,
        test=test_path,
        format='csv',
        fields=soccer_fields
    )

    COMMENT.build_vocab(train)

    train_iter, test_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, test, valid),
        batch_sizes=(2, 2, 2),
        device=-1,
        sort_within_batch=False,
        sort_key=lambda x: len(x.comment),
        repeat=False
    )
    
    return (train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    create_soccer_dataset(
        "./data/train.csv",
        "./data/test.csv",
        "./data/valid.csv"
    )
