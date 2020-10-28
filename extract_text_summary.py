"""Extract test summary script."""

from __future__ import division, print_function

import functools
import glob
import os
import time

import smile as sm
import tensorflow as tf
from smile import flags, logging


flags.DEFINE_string("event_file", "", "TF summary event file.")
flags.DEFINE_string("event_dir", "", "TF summary event dir.")
flags.DEFINE_string("tag", "", "Tag to show.")
flags.DEFINE_integer("step", 1, "Desired event step.")
FLAGS = flags.FLAGS

def show_event_file(event_file):
    try:
        it = tf.train.summary_iterator(event_file)
    except:
        logging.error("Corrupted file: " % event_file)
        return
    for event in it:
        if event.step == FLAGS.step:
            for v in event.summary.value:
                if v.tensor and v.tensor.string_val:
                    if FLAGS.tag and FLAGS.tag != v.tag:
                        continue
                    if FLAGS.tag:
                        print("\n".join(v.tensor.string_val).replace(", ", ","))
                        break
                    logging.info(v.tag)
                    logging.info("\n".join(v.tensor.string_val))

def main(_):
    """Main train script."""
    if FLAGS.event_file:
        show_event_file(FLAGS.event_file)
    if FLAGS.event_dir:
        event_files = glob.glob(
            os.path.join(FLAGS.event_dir, "events.out.tfevents*"))
        for event_file in event_files:
            try:
                show_event_file(event_file)
            except:
                if FLAGS.tag:
                    return
                logging.error("Exception occured: %s" % event_file)


if __name__ == "__main__":
    sm.app.run()