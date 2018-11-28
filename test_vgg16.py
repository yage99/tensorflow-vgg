#!/usr/bin/python3
import tensorflow as tf
import sys

import vgg16
import utils


def test(batch):
    with tf.Session(
            config=tf.ConfigProto(gpu_options=(
                tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        utils.print_prob(prob[0], './synset.txt')


if __name__ == "__main__":
    path = sys.argv[1]
    img1 = utils.load_image(path)

    batch = img1.reshape((1, 224, 224, 3))

    test(batch)
