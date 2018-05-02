import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inferenceLeNet
import mnist_trainLeNet
import numpy as np

EVAL_INTERVAL_SECS = 60


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [None, mnist_inferenceLeNet.INPUT_NODE], name='x-input')
        x = tf.placeholder(tf.float32,
                           [mnist.validation.num_examples,
                            mnist_inferenceLeNet.IMAGE_SIZE,
                            mnist_inferenceLeNet.IMAGE_SIZE,
                            mnist_inferenceLeNet.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inferenceLeNet.OUTPUT_NODE], name='y-input')

        reshape_xs = np.reshape(mnist.validation.images, (mnist.validation.num_examples,
                                     mnist_inferenceLeNet.IMAGE_SIZE,
                                     mnist_inferenceLeNet.IMAGE_SIZE,
                                     mnist_inferenceLeNet.NUM_CHANNELS))

        validata_feed = {x: reshape_xs, y_: mnist.validation.labels}
        y = mnist_inferenceLeNet.inference(x,False,None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_trainLeNet.MOVING_AVERAGE_DECAY)

        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_trainLeNet.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print([ckpt.model_checkpoint_path])
                    print('\n')
                    print(ckpt)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validata_feed)
                    print('After %s traing steps validation accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('NO checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()