# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")

FLAGS = tf.app.flags.FLAGS


def main(_):
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            image_placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='input')/255.0
            generated = model.net(image_placeholder, training=False)
            generated = tf.cast(generated, tf.uint8)
            generated = tf.squeeze(generated, [0])

            variables_to_restore = []
            for v in tf.global_variables():
                if not (v.name.startswith("vgg_16")):
                    variables_to_restore.append(v)
            variables_to_restore = [var for var in variables_to_restore if 'Adam' not in var.name]
            saver = tf.train.Saver(variables_to_restore, max_to_keep=1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            if os.path.exists('generated') is False:
                os.makedirs('generated')

            im = Image.open("img/test.jpg")
            for i in range(20):
                start_time = time.time()
                image_ = sess.run(generated, feed_dict={image_placeholder:[np.array(im)]})
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
            image_ = Image.fromarray(image_)
            image_.save('generated/res.jpg')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
