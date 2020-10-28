"""Unit tests for reader functions."""
from __future__ import division, print_function

import functools
import os
import unittest

import tensorflow as tf

from base_hparams import build_base_data_hparams
from data_utils import Vocabulary
from reader import make_generative_dataset, vectorize_smile


class TestReader(unittest.TestCase):
    """Test readers for generative drug project."""

    def setUp(self):
        """Set up function for each test."""
        self.data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../data/generative_test.atp"))

    def test_make_generative_dataset(self):
        """Test the correctness of test_make_generative_dataset."""
        dataset = make_generative_dataset(self.data_path)
        data_iter = dataset.make_one_shot_iterator().get_next()
        expected_id2smile = {
            0: "O[P]([O-])(=O)C(Cl)(Cl)[P](O)([O-])=O",
            2: "NC(CC[P](O)(O)=O)C(O)=O"
        }
        with tf.Session() as sess:
            line_id = 0
            while True:
                try:
                    data_dict = sess.run(data_iter)
                    feature = data_dict["feature"]
                    smile = data_dict["smile"]
                    self.assertEquals(feature.shape[0], 281)
                    # Assert not exceed the boundary.
                    self.assertNotEqual(smile, "CC(N)C1=CC=CC(=C1Cl)Cl")
                    # Validate the first and last smile smile.
                    if line_id in expected_id2smile:
                        self.assertEquals(smile, expected_id2smile[line_id])
                    line_id += 1
                except tf.errors.OutOfRangeError:
                    break

    def test_vectorize_smile(self):
        """Test the functionality of vectorize_smile."""
        dataset = make_generative_dataset(self.data_path)
        vocab = Vocabulary.get_default_vocab()
        data_hparams = build_base_data_hparams()
        vec_func = functools.partial(
            vectorize_smile, vocab=vocab, data_hparams=data_hparams)
        data_iter = dataset.map(vec_func).make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            line_id = 0
            while True:
                try:
                    data_dict = sess.run(data_iter)
                    seq_inputs = data_dict["seq_inputs"]
                    seq_labels = data_dict["seq_labels"]
                    # pylint: disable=no-member
                    self.assertEqual(seq_inputs.argmax(1)[0], vocab.GO_ID)
                    self.assertEqual(seq_labels[-1], vocab.EOS_ID)
                    # pylint: enable=no-member
                    self.assertEqual(seq_inputs.shape[0], seq_labels.shape[0])
                    if line_id == 0:
                        # Note the sequence length is 35 (plus a EOS symbol).
                        self.assertEqual(data_dict["seq_lens"], 36)
                    line_id += 1
                except tf.errors.OutOfRangeError:
                    break
