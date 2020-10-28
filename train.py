"""Train script."""

from __future__ import division, print_function

import functools
import os
import time

import numpy as np
import simplejson as json
import smile as sm
import tensorflow as tf
from smile import flags, logging

import base_hparams
import reader
from data_utils import Vocabulary
from models import DiscoveryModel
from reader import vectorize_smile

flags.DEFINE_string("dataset_spec", "{}", "Data csv path for training.")
flags.DEFINE_string("train_dir", "",
                    "Directory path used to store the checkpoints and summary.")
flags.DEFINE_string("data_hparams", "{}", "Data hparams JSON string.")
flags.DEFINE_string("hparams", "{}", "Model hparams JSON string.")
flags.DEFINE_integer("epochs", 10, "Total training epochs.")
flags.DEFINE_integer("steps_per_checkpoint", 200,
                     "Steps to perform test and save checkpoints.")

FLAGS = flags.FLAGS


def make_train_data(dataset_spec, vocab, data_hparams, epochs):
    """Make training and validation dataset."""
    # Make SMILE vectorization function.
    vec_func = functools.partial(
        vectorize_smile, vocab=vocab, data_hparams=data_hparams)

    # Prepare both train and val datasets.
    dataset_cls = getattr(reader, data_hparams.reader)
    datasets = dataset_cls(data_hparams, dataset_spec)()
    train_data, val_data, test_data = (datasets["train"], datasets["val"], 
                                       datasets["test"])

    buckets = data_hparams.buckets
    val_batch_size = data_hparams.val_batch_size
    seq_len_fn = lambda data: data["decoder_lens"]
    train_bucket_fn = tf.contrib.data.bucket_by_sequence_length(
        seq_len_fn, buckets, [data_hparams.batch_size] * (1 + len(buckets)))
    # For both val and test data, we do not perform bucket and batch padding.
    # The only process was padding.
    val_bucket_fn = tf.contrib.data.bucket_by_sequence_length(
        seq_len_fn, [], [val_batch_size]
        )

    # Train inputs are one shot iterator, repeating `epochs` times.
    train_inputs = train_data.map(
        vec_func, num_parallel_calls=16).apply(train_bucket_fn).repeat(
            epochs).make_one_shot_iterator().get_next()
    # Validation inputs are re-initializable iterator.
    val_inputs = val_data.map(vec_func).apply(
        val_bucket_fn).make_initializable_iterator()
    test_inputs = test_data.map(vec_func).apply(
        val_bucket_fn).make_initializable_iterator()
    return train_inputs, val_inputs, test_inputs


def train(hparams, data_hparams):
    vocab = Vocabulary.get_default_vocab(not data_hparams.skip_at_symbol)
    # Create global step variable first.

    train_data, val_data, test_data = make_train_data(
        json.loads(FLAGS.dataset_spec), vocab, data_hparams, FLAGS.epochs)
    model = DiscoveryModel(data_hparams, hparams, vocab)
    train_outputs, _, _ = model.build_train_graph(train_data)
    seq_loss_op, train_op = model.build_train_loss(train_data, train_outputs)
    with tf.control_dependencies([
        val_data.initializer, test_data.initializer]):
        _, val_ctr_smile_op, val_sampled_smiles_op = model.build_val_net(val_data.get_next())
        model.build_test_net(val_ctr_smile_op, val_sampled_smiles_op, test_data.get_next())
        

    train_summary_ops = tf.summary.merge(tf.get_collection("train_summaries"))
    val_summary_ops = tf.summary.merge(tf.get_collection("val_summaries"))
    test_summary_ops = tf.summary.merge(tf.get_collection("test_summaries"))

    stale_global_step_op = tf.train.get_or_create_global_step()
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir or None,
            save_checkpoint_steps=FLAGS.steps_per_checkpoint or None,
            log_step_count_steps=FLAGS.steps_per_checkpoint or None) as sess:
        if FLAGS.train_dir:
            summary_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
        else:
            summary_writer = None
        # step = 0
        while not sess.should_stop():
        # while step < 10:   
        #     step += 1
            stale_global_step, seq_loss, _, train_summary = sess.run(
                [stale_global_step_op, seq_loss_op, train_op, train_summary_ops])
            if summary_writer is not None:
                summary_writer.add_summary(train_summary, stale_global_step)
            # Run validation and test.
            # Trigger test events.
            if stale_global_step % FLAGS.steps_per_checkpoint == 0:
            # if True:
                try:
                    sess.run([val_data.initializer, test_data.initializer])
                    _, _ = sess.run([val_summary_ops, test_summary_ops])
                    # The monitored training session will pick up the summary 
                    # and automatically add them.
                except Exception as ex:
                    logging.error(str(ex))
                    raise
                except tf.errors.OutOfRangeError:
                    logging.info("Test finished. Continue training.")
                    continue
        logging.info("Coordinator request to stop.")


def main(_):
    """Main train script."""
    data_hparams = base_hparams.build_base_data_hparams()
    data_hparams.override_from_dict(json.loads(FLAGS.data_hparams))
    hparams = base_hparams.build_base_hparams()
    hparams.override_from_dict(json.loads(FLAGS.hparams))

    train(hparams, data_hparams)


if __name__ == "__main__":
    sm.app.run()
