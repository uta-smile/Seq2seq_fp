"""Readers for the SMILE and properties."""

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# TODO: Switch to explicit relative import.
from data_utils import sentence_to_token_ids, true_smile_tokenizer
from six.moves import range


def vectorize_smile(data_dict, vocab, data_hparams):
    """Vectorize the SMILEs and generate the sequence inputs and labels."""
    # Fix the GO symbol and shift the seq_label.
    smile = data_dict["smile"]
    tokenizer = lambda x: true_smile_tokenizer(
        x, skip_at_symbol=data_hparams.skip_at_symbol)

    if data_hparams.skip_at_symbol:
        smile = tf.regex_replace(smile, "@", "")

    def py_func_tokenize_smile(smi):
        """Return a py_func for tokenizing SMILE string in tf tensors."""
        # Extract token nums
        tokens = sentence_to_token_ids(
            smi, vocabulary=vocab, tokenizer=tokenizer)
        tokens = np.array(tokens, dtype=np.int32)
        # truncate if needed.
        if len(tokens) > (data_hparams.max_seq_len - 1):
            # Truncate the sequence with a space for EOS_ID
            tokens = tokens[:(data_hparams.max_seq_len - 1)]
        return tokens

    # Raw encode of the SMILEs.
    tokens = tf.py_func(py_func_tokenize_smile, [smile], tf.int32)
    tokens.set_shape((None, ))
    seq_len = tf.shape(tokens)[0] + 1
    # Save the seq_labels. [seq_length]
    seq_labels = tf.concat(
        [tokens, tf.constant([vocab.EOS_ID], dtype=tokens.dtype)], -1)
    # Produce inputs.
    seq_inputs = tf.concat(
        [tf.constant([vocab.GO_ID], dtype=tokens.dtype), tokens], -1)
    # One-hot each vector. -> [? (seq_length), TOK_DIM]
    seq_inputs = tf.one_hot(seq_inputs, len(vocab), dtype=tf.float32)
    # One-hot encoder inputs. -> [? (seq_length), TOK_DIM]
    encoder_inputs = tf.one_hot(tokens, len(vocab), dtype=tf.float32)
    return {
        "smile": smile,
        "decoder_lens": seq_len,
        "decoder_inputs": seq_inputs,
        "decoder_labels": seq_labels,
        "encoder_inputs": encoder_inputs
    }


#
# Dataset readers.
#


class BaseDataset(object):

    def __init__(self, data_hparams, dataset_spec):
        """Initialization of the base dataset.
        Args:
            data_hparams: A HParams instance from base_hparams.py.
            dataset_spec: A json dict object. Usually specify the dataset path.
        """
        self._data_hparams = data_hparams
        self._dataset_spec = dataset_spec

    def __call__(self):
        """Produce a tuple of training and validation tf.data.Dataset object."""
        raise NotImplementedError


class DiscoveryReader(BaseDataset):

    # Discovery Reader, by default, only reads the first SMILE string in each 
    # file.

    def _read_csv_dataset(self, dev=False):
        """Read CSV from the generative dataset."""
        train_csv_path = self._dataset_spec.get("train_csv_path", None)
        val_csv_path = self._dataset_spec.get("val_csv_path", None)
        test_csv_path = self._dataset_spec.get("test_csv_path", None)
        if train_csv_path is None or val_csv_path is None or test_csv_path is None:
            raise IOError(
                "train_csv_path or val_csv_path or test_csv_path does not exist in dataset_spec.")

        record_defaults = [tf.string]
        select_cols = [0]
        dataset_func = lambda path : tf.contrib.data.CsvDataset(
            path,
            record_defaults,
            header=False,
            select_cols=select_cols).map(lambda *args: {"smile": args[0]})

        return {
            "train": dataset_func(train_csv_path),
            "val": dataset_func(val_csv_path),
            "test": dataset_func(test_csv_path)
        }
        
    def __call__(self):
        return self._read_csv_dataset()
