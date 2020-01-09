from __future__ import print_function
import tensorflow as tf
import model
import time
import os
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', "")
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")
tf.app.flags.DEFINE_string("save_file", "a.jpg", "")
tf.app.flags.DEFINE_float("alpha", 1.0, "")
tf.app.flags.DEFINE_float("beta", 5, "")

FLAGS = tf.app.flags.FLAGS


def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image


def main(_):
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            image_placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='input')
            generated = model.net(image_placeholder, FLAGS=FLAGS, training=False)
            generated = tf.cast(generated, tf.uint8)
            generated = tf.squeeze(generated, [0])

            variables_to_restore = []
            for v in tf.global_variables():
                if not (v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            variables_to_restore = [var for var in variables_to_restore if 'Adam' not in var.name]
            saver = tf.train.Saver(variables_to_restore, max_to_keep=1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            if os.path.exists('generated') is False:
                os.makedirs('generated')

            im = Image.open(FLAGS.image_file)
            im = center_crop(im, 512, 512)
            start_time = time.time()
            image_ = sess.run(generated, feed_dict={image_placeholder: [np.array(im)]})
            end_time = time.time()
            image_ = Image.fromarray(image_)
            image_ = center_crop(image_, 512, 512)
            image_.save(FLAGS.save_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
