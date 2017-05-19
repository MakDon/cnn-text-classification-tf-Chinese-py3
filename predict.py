#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle
import codecs

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1494803898/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# CHANGE THIS: Load data. Load your own data here
vocabulary=pickle.load(open(os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "..", "vocab.txt")),"rb"))
sequence_length=pickle.load(open(os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "..", "len.txt")),"rb"))

class Predictor(object):
    def __init__(self):
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

    def predict(self, x_raw):
        x_raw = list(x_raw)
        x_raw = [s.strip() for s in x_raw]
        x_raw = [list(s) for s in x_raw]
        x_pad,_ = data_helpers.pad_sentences(x_raw,sequence_length)
        x_test = np.array([[vocabulary.get(word,0) for word in sentence] for sentence in x_pad])

        # Get the placeholders from the graph by name
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        return all_predictions

#test predict
#========================================
if __name__ == '__main__':
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
    x_raw = (u"小嘴喳喳里面的玩具！",)
    predictor = Predictor()
    print(predictor.predict(x_raw))