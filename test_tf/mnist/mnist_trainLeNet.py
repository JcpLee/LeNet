import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inferenceLeNet
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.22
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0005
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = '/path/to/model2/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    # x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    # y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                       mnist_inferenceLeNet.IMAGE_SIZE,
                       mnist_inferenceLeNet.IMAGE_SIZE,
                       mnist_inferenceLeNet.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None, mnist_inferenceLeNet.OUTPUT_NODE],
                        name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inferenceLeNet.inference(input_tensor=x,train=True,regularizer=regularizer)

    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs,(BATCH_SIZE,
                                        mnist_inferenceLeNet.IMAGE_SIZE,
                                        mnist_inferenceLeNet.IMAGE_SIZE,
                                        mnist_inferenceLeNet.NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs,y_:ys})
            if i%1000 == 0:
                print('After %d training steps,loss on training batch is %g'%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
        # xa,ya = mnist.train.images,mnist.train.labels
        # _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xa, y_: ya})
        # print('After %d training steps,loss on training set is %g' % (step, loss_value))
def main(argv = None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
