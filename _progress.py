'''
## on server ##

extract the 4 attributes result with data

using 4 learned checkpoints

'''

import tensorflow as tf
import time as t
import os
import shutil


ckpt_list = ["/home/ubuntu/hoon/ckpts/shape/ShapeCkpt-60",
             "/home/ubuntu/hoon/ckpts/size/SizeCkpt-51",
             "/home/ubuntu/hoon/ckpts/type/TypeCkpt-50",
             "/home/ubuntu/hoon/ckpts/color/TypeCkpt-85"]


image_height = 224
image_width = 224
num_out = 3

keep_prob = tf.placeholder(dtype=tf.float32)
task = tf.placeholder(dtype=tf.bool)
x = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, 3])
z = tf.placeholder(dtype=tf.float32, shape=[None, 1]) # for gender


# create weight function
def weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# create bias function
def bias(shape, num, name):
    if num == 0.0: # conv-layer : initialie to 0.0
        initial = tf.zeros(shape, dtype=tf.float32)
    else: # fully-connected layer : initialize to 1.0
        initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# conv2d wrapping function
def conv(x, y):
    return tf.nn.conv2d(x, y, strides=[1,1,1,1], padding="SAME")


# batch_normalization function for conv-layer
def batch_norm(batch_data, n_out, is_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(batch_data, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(batch_data, mean, var, beta, gamma, 1e-3)
    return normed

# batch_normalization function for fully-connected layer
def batch_FC(inputs, is_train):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    batch_mean, batch_var = tf.nn.moments(inputs, [0])
    ema2 = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
        ema2_apply_op = ema2.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema2_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema2.average(batch_mean), ema2.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, scale, 1e-3)
    return normed


# convolution layers
w_conv1_1 = weight([3,3,3,64], 'w_conv1_1')
b_conv1_1 = bias([64], 0.0, 'b_conv1_1')
w_conv1_2 = weight([3,3,64,64], 'w_conv1_2')
b_conv1_2 = bias([64], 0.0, 'b_conv1_2')
w_conv2_1 = weight([3,3,64,128], 'w_conv2_1')
b_conv2_1 = bias([128], 0.0, 'b_conv2_1')
w_conv2_2 = weight([3,3,128,128], 'w_conv2_2')
b_conv2_2 = bias([128], 0.0, 'b_conv2_2')
w_conv3_1 = weight([3,3,128,256], 'w_conv3_1')
b_conv3_1 = bias([256], 0.0 , 'b_conv3_1')
w_conv3_2 = weight([3,3,256,256], 'w_conv3_2')
b_conv3_2 = bias([256], 0.0, 'b_conv3_2')
w_conv3_3 = weight([3,3,256,256], 'w_conv3_3')
b_conv3_3 = bias([256], 0.0, 'b_conv3_3')
w_conv4_1 = weight([3,3,256,512], 'w_conv4_1')
b_conv4_1 = bias([512], 0.0, 'b_conv4_1')
w_conv4_2 = weight([3,3,512,512], 'w_conv4_2')
b_conv4_2 = bias([512], 0.0, 'b_conv4_2')
w_conv4_3 = weight([3, 3, 512, 512], 'w_conv4_3')
b_conv4_3 = bias([512], 0.0, 'b_conv4_3')
w_conv5_1 = weight([3,3,512,512], 'w_conv5_1')
b_conv5_1 = bias([512], 0.0, 'b_conv5_1')
w_conv5_2 = weight([3,3,512,512], 'w_conv5_2')
b_conv5_2 = bias([512], 0.0, 'b_conv5_2')
w_conv5_3 = weight([3,3,512,512], 'w_conv5_3')
b_conv5_3 = bias([512], 0.0, 'b_conv5_3')

# fully connected layers
w_fc1 = weight([7*7*512, 4096], 'w_fc1')
b_fc1 = bias([4096], 1.0, 'b_fc1')
w_fc2 = weight([4096, 4096], 'w_fc2')
b_fc2 = bias([4096], 1.0, 'b_fc2')
w_vgg = weight([4096, num_out], 'w_vgg')
b_vgg = bias([num_out], 1.0, 'b_vgg')
w_gender1 = weight([num_out + 1, 12], 'w_gender1')
b_gender1 = bias([12], 1.0, 'b_gender1')
w_gender2 = weight([12, num_out], 'w_gender2')
b_gender2 = bias([num_out], 1.0, 'b_gender2')

#x_image = tf.reshape(x, shape=[-1, 224, 224, 3])
y_label = tf.reshape(y, shape=[-1, num_out])
z_gender = tf.reshape(z, shape=[-1, 1])

conv1_1 = tf.nn.relu(batch_norm((conv(x, w_conv1_1) + b_conv1_1), 64, task))
conv1_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv1_1, w_conv1_2), b_conv1_2), 64, task))
pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool1, w_conv2_1), b_conv2_1),128, task))
conv2_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv2_1, w_conv2_2), b_conv2_2), 128, task))
pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv3_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool2, w_conv3_1), b_conv3_1), 256, task))
conv3_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv3_1, w_conv3_2), b_conv3_2), 256, task))
conv3_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv3_2, w_conv3_3), b_conv3_3), 256, task))
pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv4_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool3, w_conv4_1), b_conv4_1), 512, task))
conv4_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv4_1, w_conv4_2), b_conv4_2), 512, task))
conv4_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv4_2, w_conv4_3), b_conv4_3), 512, task))
pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv5_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool4, w_conv5_1), b_conv5_1), 512, task))
conv5_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv5_1, w_conv5_2), b_conv5_2), 512, task))
conv5_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv5_2, w_conv5_3), b_conv5_3), 512, task))
pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
fc1 = tf.nn.relu(batch_FC(tf.nn.dropout(tf.nn.bias_add(tf.matmul(flat, w_fc1), b_fc1), keep_prob=keep_prob), task))
fc2 = tf.nn.relu(batch_FC(tf.nn.dropout(tf.nn.bias_add(tf.matmul(fc1, w_fc2), b_fc2), keep_prob=keep_prob), task))
y_vgg = tf.nn.dropout(tf.nn.bias_add(tf.matmul(fc2, w_vgg), b_vgg), keep_prob=keep_prob)
y_vgg = tf.concat([y_vgg, z_gender], 1)
y_gender = tf.nn.relu(batch_FC(tf.nn.dropout(tf.nn.bias_add(tf.matmul(y_vgg, w_gender1), b_gender1), keep_prob=keep_prob), task))
y_out = tf.nn.dropout(tf.nn.bias_add(tf.matmul(y_gender, w_gender2), b_gender2), keep_prob=keep_prob)

max_point = tf.cast(tf.argmax(y_out, 1), dtype=tf.int32)

global_step = tf.Variable(0, trainable=False)
step = 1

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=22)
    while True:
        try:
            f = open("data.txt", 'r')
        except:
            print("sleep")
            t.sleep(0.5)
        else:
            start_time = t.time()
            data = f.readline()
            print(data)
            shutil.copy(data, "/home/ubuntu/opt/pyshell/save/"+str(step)+".jpg")
            step = step + 1
            f.close()
            os.remove("data.txt")
            image_position = data
            start_time = t.time()
            pre_img_value = tf.read_file(data)
            img_value = tf.reshape(tf.cast(tf.image.decode_jpeg(pre_img_value, channels=3), dtype=tf.float32), shape=[1, image_height, image_width, 3])
            x_image = sess.run(img_value)
            glasses = [-1, -1, -1, -1]
            for attribute in range(4):
                saver.restore(sess, ckpt_list[attribute])
                result = sess.run(max_point, feed_dict={keep_prob: 1.0, x: x_image, task: False})
                if attribute < 2:
                    if result[0] < 2:
                        glasses[attribute] = result[0]
                    else:
                        glasses[attribute] = 1
                else:
                    glasses[attribute] = result[0]
            f2 = open("output.txt", "w")
            for attribute in range(4):
                f2.write(str(glasses[attribute]))
            f2.close()

            print(glasses)
            b = t.time()
            print(str(b-start_time) + " second")
