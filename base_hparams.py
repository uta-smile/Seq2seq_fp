"""Contains the base data and model hparams."""

from __future__ import division, print_function

import tensorflow as tf


def build_base_data_hparams():
    """Create a base data hparams instance."""
    return tf.contrib.training.HParams(
        # Reader class name.
        reader="GenerativeReader",
        # Max length of the input sequences.
        max_seq_len=120,
        # Training batch size.
        batch_size=32,
        # Val batch size. (How many samples will be processed in each iter.)
        val_batch_size=32,
        # Val data num. (How many samples in total in val data.)
        val_data_num=10000,
        # Buckets.
        buckets=[30, 60],
        # Skip at symobl. If we skip "@" symbol in smile.
        skip_at_symbol=True)


def build_base_hparams():
    return tf.contrib.training.HParams(
        # RNN Cell Size.
        cell_size=128,
        # Input/output embed dimention, zero for not embedding.
        # > 0 is not working right now. do not use.
        embed_dim=0,
        # Start learning rate.
        init_learning_rate=5e-3,
        # Hit trials. [100, 200, 500] indicates that to test 
        # the max TM hit at 100, 200, 500.
        search_hits=[100, 200, 500])
