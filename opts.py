#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : options from opennmt-py
"""


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Device
    group = parser.add_argument_group('Device')
    group.add_argument('-device',
                       default="cpu",
                       help="e.g., cpu or cuda:1")

    # Data options
    group = parser.add_argument_group('Data')

    group.add_argument('-resources_dir',
                       default='~/common_corpus/',
                       help="where wiki.. glove are")
    group.add_argument('-vector_type',
                       default="glove.6B.100d",
                       help="E.g., glove.6b.300d")
    group.add_argument('-batch_size', type=int,
                       default=20,
                       help="batch size")

    # Model options
    group = parser.add_argument_group('Model')
    group.add_argument(
        '-input_embeddings_trainable',
        default=1,
        type=int,
        help="whether train inputembeddings 1 to train 0 to not train"
    )
    group.add_argument('-save', default="nnlm.model", help="the saving path")
    group.add_argument(
        '-every_n_epoch_save',
        default=4,
        type=int,
        help="every this epoch saving model"
    )
    group.add_argument('-seed', default=0, help="random seed", type=int)
    group.add_argument(
        '-tied', default=True,
        help="tied input and output embedding",
        type=bool
    )
    group.add_argument('-epoch', default=30, help="epoch", type=int)
    group.add_argument('-lr', default=0.1, help="learning rate", type=float)
    group.add_argument('-rnn_type', default='GRU', help="type", type=str)
    group.add_argument('-bidirectional', default=False, type=bool)
    group.add_argument(
        '-num_layers',
        default=1,
        type=int,
        help="number of layers"
    )

    group.add_argument(
        '-dropout', default=0.0, help="dropout", type=float
    )
    group.add_argument(
        '-hidden_size', default=100, help="hidden_size", type=int
    )
    group.add_argument(
        '-src_wd_dim', default=50, help="src_wd_dim", type=int
    )

    group = parser.add_argument_group('Logging')
    group.add_argument('-log_file', type=str, default="log",
                       help="Output logs to a file under this path.")
