import tensorflow as tf
import config.config as cfg
from core.conv_def import conv_bn_relu,deconv_bn_relu,conv2d
from core.loss_func import *

class Unet():
    def __init__(self,is_training):
        self.is_training = is_training
        self.loss_coefficient = 1e4
        # predefined
        # single-gpu
        self.gpu_number = len(cfg.gpu.split(','))
        if self.gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']

        self.output_channels = 2

    def inference_op(self,_input):

        concat_dimension = -1  # channels_last
        # padding output
        with tf.device(device_name_or_function=self.device[0]):
            # first level
            encoder1_1 = conv_bn_relu(inputs=_input, output_channels=16, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder1_1')
            encoder1_2 = conv_bn_relu(inputs=encoder1_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder1_2')
            pool1 = tf.layers.max_pooling2d(
                inputs=encoder1_2,
                pool_size=2,                    # pool_depth, pool_height, pool_width
                strides=2,
                padding='valid',                # No padding, default
                data_format='channels_last',    # default
                name='pool1'
            )
            # second level
            encoder2_1 = conv_bn_relu(inputs=pool1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder2_1')
            encoder2_2 = conv_bn_relu(inputs=encoder2_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder2_2')
            pool2 = tf.layers.max_pooling2d(inputs=encoder2_2, pool_size=2, strides=2, name='pool2')
            # third level
            encoder3_1 = conv_bn_relu(inputs=pool2, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder3_1')
            encoder3_2 = conv_bn_relu(inputs=encoder3_1, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder3_2')
            pool3 = tf.layers.max_pooling2d(inputs=encoder3_2, pool_size=2, strides=2, name='pool3')
            # forth level
            encoder4_1 = conv_bn_relu(inputs=pool3, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder4_1')
            encoder4_2 = conv_bn_relu(inputs=encoder4_1, output_channels=256, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder4_2')
            bottom = encoder4_2

        # up-sampling path
        # device: gpu1
        with tf.device(device_name_or_function=self.device[1]):
            # third level
            deconv3 = deconv_bn_relu(inputs=bottom, output_channels=256, is_training=self.is_training,
                                     name='deconv3')
            concat_3 = tf.concat([deconv3, encoder3_2], axis=concat_dimension, name='concat_3')
            decoder3_1 = conv_bn_relu(inputs=concat_3, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder3_1')
            decoder3_2 = conv_bn_relu(inputs=decoder3_1, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder3_2')
            # second level
            deconv2 = deconv_bn_relu(inputs=decoder3_2, output_channels=128, is_training=self.is_training,
                                     name='deconv2')
            concat_2 = tf.concat([deconv2, encoder2_2], axis=concat_dimension, name='concat_2')
            decoder2_1 = conv_bn_relu(inputs=concat_2, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder2_1')
            decoder2_2 = conv_bn_relu(inputs=decoder2_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder2_2')
            # first level
            deconv1 = deconv_bn_relu(inputs=decoder2_2, output_channels=64, is_training=self.is_training,
                                     name='deconv1')
            concat_1 = tf.concat([deconv1, encoder1_2], axis=concat_dimension, name='concat_1')
            decoder1_1 = conv_bn_relu(inputs=concat_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder1_1')
            decoder1_2 = conv_bn_relu(inputs=decoder1_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder1_2')
            feature = decoder1_2
            # predicted probability
            predicted_prob = conv2d(inputs=feature, output_channels=self.output_channels, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_prob')

        with tf.variable_scope('prediction'):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=-1, name='predicted_label')

        return predicted_prob,predicted_label

    def loss_op(self, logits, labels):

        with tf.name_scope("jaccard_loss"):
            #self.total_loss = jacc_loss_new(logits[0], labels)
            labels = tf.cast(labels, dtype=tf.int64)
            self.total_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[0],labels=labels))
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(logits[1], tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar("total_loss", self.total_loss)

        return self.total_loss, accuracy

