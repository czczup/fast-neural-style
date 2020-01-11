# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os

import model
import utils


class FLAGS:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


def export(ckpt_file, pb_file, FLAGS):
    g = tf.Graph()      # A new graph
    with g.as_default():
        with tf.Session() as sess:
            image_placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='input')

            generated_image = model.net(image_placeholder, FLAGS=FLAGS, training=False)

            casted_image = tf.cast(generated_image, tf.int8)
            # Remove batch dimension
            squeezed_image = tf.squeeze(casted_image, [0], name='output')
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            saver.restore(sess, ckpt_file)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=['output'])

            with tf.gfile.FastGFile(pb_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())


export("model/fast-style-model.ckpt-93000", "model/fast-style-model.pb", FLAGS=FLAGS(alpha=1.0, beta=5))