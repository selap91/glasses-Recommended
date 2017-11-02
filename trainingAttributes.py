import tensorflow as tf
import time as t
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"  # graphic cared number to use

# data csv files
train_csv_dir = "/mnt/hdd3t/Data/hci1/hoon/LightHouse_of_Inha/CSVs/3th/size/train_G_size.csv"
test_csv_dir = "/mnt/hdd3t/Data/hci1/hoon/LightHouse_of_Inha/CSVs/3th/size/test_G_size.csv"

image_height = 224
image_width = 224
train_batch_size = 32 # batch size
test_batch_size = 15
num_out = 2 # number of output node

keep_prob = tf.placeholder(dtype=tf.float32) # drop-out %
task = tf.placeholder(dtype=tf.bool) # true : training / false : testing
x = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, 3]) # for image
y = tf.placeholder(dtype=tf.float32, shape=[None, num_out]) # for label
z = tf.placeholder(dtype=tf.float32, shape=[None, 1]) # for gender


# train data load
train_queue = tf.train.string_input_producer([train_csv_dir])
train_reader = tf.TextLineReader()
_, train_csv_value = train_reader.read(train_queue)
train_img_dir, train_label, train_gender = tf.decode_csv(train_csv_value, record_defaults=[[""], [-1], [-1]])
train_img_value = tf.read_file(train_img_dir)
train_img = tf.reshape(tf.cast(tf.image.decode_jpeg(train_img_value, channels=3), dtype=tf.float32), shape=[image_height, image_width, 3])
train_label = tf.reshape(tf.one_hot(train_label, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
train_gender = tf.reshape(train_gender, shape=[1])

# test data load
test_queue = tf.train.string_input_producer([test_csv_dir], shuffle=False)
test_reader = tf.TextLineReader()
_, test_csv_value = test_reader.read(test_queue)
test_img_dir, test_label, test_gender = tf.decode_csv(test_csv_value, record_defaults=[[""], [-1], [-1]])
test_img_value = tf.read_file(test_img_dir)
test_img = tf.reshape(tf.cast(tf.image.decode_jpeg(test_img_value, channels=3), dtype=tf.float32), shape=[image_height, image_width, 3])
test_label = tf.reshape(tf.one_hot(test_label, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
test_gender = tf.reshape(test_gender, shape=[1])


def weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias(shape, num, name):
    if num == 0.0:
        initial = tf.zeros(shape, dtype=tf.float32)
    else:
        initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

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
y_out = tf.nn.dropout(tf.nn.bias_add(tf.matmul(fc2, w_vgg), b_vgg), keep_prob=keep_prob)
#y_gender = tf.concat([y_vgg, z_gender], 1)
#y_out = tf.nn.bias_add(tf.matmul(y_gender, w_gender), b_gender)
#y_1 = tf.nn.dropout(tf.nn.relu(batch_FC(tf.nn.bias_add(tf.matmul(y_gender, w_gender), b_gender), 0.997, task)), keep_prob=keep_prob)
#y_out = tf.nn.bias_add(tf.matmul(y_1, w_out), b_out)

label_value = tf.reshape(tf.cast(tf.argmax(y_label, 1), dtype=tf.int32), shape=[test_batch_size])
#vgg_max = tf.reshape(tf.cast(tf.argmax(y_vgg, 1), dtype=tf.int32), shape=[test_batch_size])
max_point = tf.reshape(tf.cast(tf.argmax(y_out, 1), dtype=tf.int32), shape=[test_batch_size])

# train
start_learning_rate = 0.0008 # start_learning_rate
global_step = tf.Variable(0, trainable=False)
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=y_out))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_out))
tf.summary.scalar('loss', cross_entropy)
learning_rate = tf.maximum(0.00003, tf.train.exponential_decay(start_learning_rate, global_step, 467, 0.8, staircase=True))
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

# accuracy
prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)



b_train_image, b_train_label, b_train_dir, b_train_gender = tf.train.shuffle_batch([train_img, train_label, train_img_dir, train_gender], batch_size=train_batch_size, num_threads=1, capacity=10000, min_after_dequeue=0, allow_smaller_final_batch=True)
b_test_image, b_test_label, b_test_dir, b_test_gender = tf.train.batch([test_img, test_label, test_img_dir, test_gender], batch_size=test_batch_size, num_threads=1, capacity=10000, allow_smaller_final_batch=True)


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=22)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, '/mnt/hdd3t/Data/hci1/hoon/2th/shape/total/2thCircleCkpts/CircleCkpt-50')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./3thSizeSummaries/', sess.graph)
    now_epoch = 0

    for i in range(18702):
        if i % 935 != 0:
            batch_x, batch_y, batch_z = sess.run([b_train_image, b_train_label, b_train_gender])
            sess.run(train, feed_dict={keep_prob: 0.5, x: batch_x, y: batch_y, z: batch_z, task: True})

        elif i % 935 == 0:
            batch_x, batch_y, batch_d, batch_z = sess.run([b_train_image, b_train_label, b_train_dir, b_train_gender])
            _, acc, loss, mer, lr = sess.run([train, accuracy, cross_entropy, merge, learning_rate], feed_dict={keep_prob: 0.5, x: batch_x, y: batch_y, z: batch_z, task: True})
            print("epoch  ", now_epoch)
            print("   LR = ", lr)
            print("   loss = ", loss)
            print("   accuracy = " + str(acc*100) + " %")
            currentTime = t.localtime()
            print(t.strftime("%Y-%m-%d %H:%M:%S", currentTime))
            train_writer.add_summary(mer, now_epoch)
            saver.save(sess, './3thSizeCkpts/SizeCkpt', global_step=now_epoch)
            print("--------------------------------------------------------------------------------")

            t_acc = 0.0
            t_acc2 = 0.0
            tAcc2= 0.0
            for j in range(54):
                test_batch_x, test_batch_y, test_batch_d, test_batch_z = sess.run([b_test_image, b_test_label, b_test_dir, b_test_gender])
                t_acc = 0.0
                count_list = [0, 0]
                tAcc, result, labels = sess.run([accuracy, max_point, label_value], feed_dict={keep_prob: 1.0, x: test_batch_x, y: test_batch_y, z: test_batch_z, task: False})
                for g in range(len(result)):
                    if result[g] == 0:
                        count_list[0] = count_list[0] + 1
                    else:
                        count_list[1] = count_list[1] + 1

                t_acc = float(float(max(count_list))/float(len(result)))
                t_acc2 = t_acc2 + t_acc
                tAcc2 = tAcc2 + tAcc

                if now_epoch % 20 == 0:
                    # print(result2)
                    print(result)
                    print(labels)
                    print(str(j) + " Test Data Accuracy = %-4.2f // Match Percent = %-4.2f" % (t_acc * 100, tAcc * 100))

            print("Total Test Data Accuracy = %-4.2f // Original Accuracy = %-4.2f" % (t_acc2/54*100, tAcc2/54*100))
            now_epoch = now_epoch + 5
            print("========================================================================================")
            print("========================================================================================")

    coord.request_stop()
    coord.join(threads)



