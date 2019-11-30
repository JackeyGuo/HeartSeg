import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import config.config as cfg
from core.layers import _conv2d, _active_layer, deconv2d, conv_bn_prelu, \
    deconv_bn_prelu, atrous_bn_prelu, BilinearUpsample2d, bn_prelu_conv,crop_tensor
from core.loss_func import *


class xception_aspp(object):
    def __init__(self, is_training):

        self.layer_stack = []
        self.cont_block_num = 3
        self.expand_block_num = 3
        #self.loss_coefficient = 1e4
        self.is_training = is_training
        self.atrou_num = 3

        self.gpu_number = len(cfg.gpu.split(','))
        if self.gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']
        self.output_channels = {'1': [16, 16, 16], '2': [32, 32, 32], '3': [64, 64, 64], '4': [32, 32], '5': [32, 16],'6': [16, 8]}
        self.stride = {'1': [2], '2': [4], '3': [8]}
        self.output_classes = cfg.classes

        self.class_layer = []

    def feature_pyramid(self,_input,block_num):
        atrous_layer = []
        out_feature = int(_input.get_shape()[-1])

        output_1x1 = _conv2d(_input, kernel_size=1, stride=1, output_feature=out_feature, use_bias=True, name='pyramid_conv_1')

        for i in range(1, self.atrou_num + 1):
            dilate_rate = int(np.power(2,4-block_num)*i)
            #dilate_rate = int(2*(4 - block_num) * i)
            #print(dilate_rate)
            # dilate rate : 24 16 8 / 12 8 4 / 6 4 2
            # dilate rate : 18 12 6 / 12 8 4 / 6 4 2
            output = atrous_bn_prelu(_input, kernel_size=3, stride=1, output_channels=out_feature/2,
                             dilation_rate=dilate_rate, is_training=self.is_training,name='atrous_conv%d' % i)

            atrous_layer.append(output)

        output = tf.concat([output_1x1, atrous_layer[0], atrous_layer[1], atrous_layer[2]], axis=-1)
        #print('atrous conv shape:', output)
        output = _conv2d(output, kernel_size=1, stride=1, output_feature=out_feature,use_bias=True, name='pyramid_conv_1x1')

        return output
    def cascade_feature(self,_input,block_num):
        atrous_layer = []
        out_feature = int(_input.get_shape()[-1])
        output = _input
        #output_1x1 = _conv3d(_input, kernel_size=1, stride=1, output_feature=out_feature, use_bias=True, name='pyramid_conv_1')

        for i in range(1, self.atrou_num + 1):
            dilate_rate = int(np.power(2,4-block_num)*i)
            #print(dilate_rate)
            # dilate rate : 24 16 8 / 12 8 4 / 6 4 2
            output = atrous_bn_prelu(output, kernel_size=3, stride=1, output_channels=out_feature,
                             dilation_rate=dilate_rate, is_training=self.is_training,name='atrous_conv%d' % i)

            atrous_layer.append(output)

        output = tf.concat([atrous_layer[0], atrous_layer[1], atrous_layer[2]], axis=-1)
        #print('atrous conv shape:', output)
        output = _conv2d(output, kernel_size=1, stride=1, output_feature=out_feature,use_bias=True, name='pyramid_conv_1x1')

        return output
    def _contracting_block(self, block_num, _input, input_layer):
        # xception contracte blcok
        output = conv_bn_prelu(_input, kernel_size=3, stride=1, output_channels=self.output_channels[str(block_num)][0],
                         name='conv_1', is_training=self.is_training)

        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=self.output_channels[str(block_num)][1],
                               name='conv_2', is_training=self.is_training)

        output = conv_bn_prelu(output, kernel_size=3, stride=2, output_channels=self.output_channels[str(block_num)][2],
                         name='conv_3', is_training=self.is_training)
        #output = self.feature_pyramid(output,block_num)
        #output = self.cascade_feature(output, block_num)
        #print('output',output)
        # do conv 1 x 1 x 1 before sum
        #for i in range(block_num):
            #input_layer = crop_tensor(input_layer, block_num)
        #input = _conv3d(input, kernel_size=1, stride=1, output_feature=self.output_channels[str(block_num)][2], use_bias=True, name='conv_s2')
        input = conv_bn_prelu(input_layer, kernel_size=3, stride=self.stride[str(block_num)][0], output_channels=self.output_channels[str(block_num)][2],
                               name='conv_s2', is_training=self.is_training)
        #input = tf.concat([input, input],axis=-1)
        #logits_slice = tf.slice(input_layer, [0, half_receptive, half_receptive, half_receptive, 0],
                                #output.get_shape()
        #print(input)
        output = tf.add(output, input, name='elemwise_sum')

        output = self.feature_pyramid(output,block_num)
        #output = self.cascade_feature(output, block_num)
        print('output',output)
        return output

    def _expanding_block(self, block_num, _input, layer, option='concat'):
        output = _conv2d(_input, kernel_size=1, stride=1, output_feature=self.output_channels[str(block_num)][0],
                         use_bias=True, name='conv_1')

        output = deconv_bn_prelu(output, output_channels=self.output_channels[str(block_num)][1],
                                 is_training=self.is_training, name='deconv')

        if option == 'sum':
            output = tf.add(output, layer, name='elemwise_sum')
        else:
            output = tf.concat(values=(output, layer), axis=-1, name='concat')
        # 26*50*40*64 / 52*100*80*32 / 104*200*160*16
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=int(output.get_shape()[-1]),
                               name='conv_2', is_training=self.is_training)
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=int(output.get_shape()[-1]),
                              name='conv_3', is_training=self.is_training)
        return output

    def inference_op(self, _input):

        conv_layer = []
        dconv_layer = []
        # padding output
        #output = tf.pad(_input, np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0]]), name='pad_1')
        with tf.device(device_name_or_function=self.device[0]):
            output = conv_bn_prelu(_input, output_channels=8, kernel_size=3, stride=1,
                                   is_training=self.is_training, name='conv_1') # 104x200x160 8
            input_layer = output
            conv_layer.append(output)

            for block_num in range(1, self.cont_block_num + 1):
                with tf.variable_scope('contract_block_%d' % block_num):
                    output = self._contracting_block(block_num,output,input_layer)
                    conv_layer.append(output)
            #self.class_layer.append(conv_layer[-1])
            self.class_layer.append(conv_layer[-1])

            for block_num in range(4, self.expand_block_num + 4):
                with tf.variable_scope('expand_block_%d' % block_num):
                    output = self._expanding_block(block_num, output, conv_layer[2 - block_num])
                    dconv_layer.append(output)
        with tf.device(device_name_or_function=self.device[1]):

            '''auxiliary prediction'''

            # forth level
            auxiliary3_prob_4x = _conv2d(inputs=dconv_layer[0], output_feature=self.output_classes, kernel_size=1,
                                         stride=1, use_bias=True, name='auxiliary3_prob_4x')
            auxiliary3_prob_2x = deconv2d(inputs=auxiliary3_prob_4x, output_channels=self.output_classes,
                                          name='auxiliary3_prob_2x')
            auxiliary3_prob_1x = deconv2d(inputs=auxiliary3_prob_2x, output_channels=self.output_classes,
                                          name='auxiliary3_prob_1x')
            # third level
            auxiliary2_prob_2x = _conv2d(inputs=dconv_layer[1], output_feature=self.output_classes, kernel_size=1,
                                         stride=1, use_bias=True, name='auxiliary2_prob_2x')
            auxiliary2_prob_1x = deconv2d(inputs=auxiliary2_prob_2x, output_channels=self.output_classes,
                                          name='auxiliary2_prob_2x')
            # second level
            auxiliary1_prob_1x = _conv2d(inputs=dconv_layer[2], output_feature=self.output_classes, kernel_size=1,
                                         stride=1, use_bias=True, name='auxiliary1_prob_1x')

            with tf.variable_scope('last_stage'):
                # out_feature = int(output.get_shape()[-1]) / 2
                #print(dconv_layer[0],dconv_layer[1],dconv_layer[2])
                output1 = _conv2d(dconv_layer[0], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                                  name='block1_conv1x1')

                #output1 = deconv3d(output1, output_channels=int(output1.get_shape()[-1]), name='block1_deconv')
                output1 = BilinearUpsample2d(output1, up_factor=2)

                output2 = _conv2d(dconv_layer[1], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                                   name='block2_conv1x1')

                output2 = tf.add(output1, output2)

                #output2 = deconv3d(output2, output_channels=int(output2.get_shape()[-1]), name='block2_deconv')
                output2 = BilinearUpsample2d(output2, up_factor=2)

                output3 = _conv2d(dconv_layer[2], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                                   name='block3_conv1x1')

                output3 = tf.add(output2, output3)

                output = _conv2d(output3, kernel_size=1, stride=1, output_feature=self.output_classes, use_bias=True, name='fc_layer')

        with tf.device(device_name_or_function=self.device[2]):
            with tf.variable_scope('prediction'):
                softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
                predicted_label = tf.argmax(input=softmax_prob, axis=-1, name='predicted_label')

        if cfg.joint_train:
            with tf.variable_scope('class_layer'):
                cls_input = output
                cls_outputs = _conv2d(cls_input, kernel_size=3, stride=1, output_feature=int(cls_input.get_shape()[-1]), name='cls_conv1')
                # average pooling
                last_pool_kernel = int(cls_outputs.get_shape()[-2])
                k = last_pool_kernel
                cls_outputs = tf.nn.avg_pool(cls_outputs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
                # print(k,outputs)
                # FC
                features_total = int(cls_outputs.get_shape()[-1])
                cls_outputs = tf.reshape(cls_outputs, [-1, features_total])
                W = tf.get_variable(shape=[features_total, 2], initializer=tf.contrib.layers.xavier_initializer(), name='W')
                bias = tf.get_variable(initializer=tf.constant(0.0, shape=[2]),name='bias')
                cls_outputs = tf.matmul(cls_outputs, W) + bias

            with tf.variable_scope('class_prediction'):
                cls_softmax = tf.nn.softmax(logits=cls_outputs, name='softmax_prob')
                cls_predict = tf.argmax(input=cls_softmax, axis=-1, name='predicted_label')
            return output, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x, cls_outputs, cls_predict
        else:
            return output, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def class_loss(self, outputs, labels):

        cls_outputs = outputs[-2]
        # softmax_prediction = tf.nn.softmax(logits=cls_outputs)
        # ground_truth = tf.one_hot(labels, depth=2)
        # class_loss = 0
        # for i in range(2):
        #     class_i_ground_truth = ground_truth[:, i]
        #     class_i_prediction = softmax_prediction[:, i]
        #     weighted = (1 - tf.reduce_mean(class_i_ground_truth)) / tf.reduce_mean(class_i_ground_truth)
        #
        #     class_loss = class_loss + tf.reduce_mean(
        #         tf.nn.weighted_cross_entropy_with_logits(logits=class_i_prediction, targets=class_i_ground_truth, pos_weight=weighted))
        #     # class_loss = class_loss - tf.reduce_mean(
        #     #     weighted * class_i_ground_truth *tf.log(tf.clip_by_value(class_i_prediction, 1e-8, 1.0)) +
        #     #         (1 - class_i_ground_truth) * tf.log(tf.clip_by_value(1-class_i_prediction, 1e-8, 1.0)))
        #         # Clips tensor values to a specified min and max.
        #     tf.summary.scalar("pos_weight", tf.reduce_mean(weighted))
        # compute weights
        # pw = (1 - tf.reduce_mean(labels)) / tf.reduce_mean(labels)
        # self.class_loss = tf.reduce_mean(
        #      tf.nn.weighted_cross_entropy_with_logits(logits=cls_outputs, targets=labels, pos_weight=pw))
        labels = tf.one_hot(labels, depth=2)
        self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_outputs,labels=labels))

        with tf.name_scope("accuracy"):
            #self.prediction = tf.nn.softmax(cls_outputs)
            correct_pred = tf.equal(outputs[-1], tf.argmax(labels,axis=-1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('class accuracy', accuracy)
        tf.summary.scalar('class loss', self.class_loss)

        return self.class_loss, accuracy

    def loss_op(self, outputs, labels, dst_weight=None):
        logits = outputs[0]
        pred_labels = outputs[1]
        auxiliary1_prob_1x = outputs[2]
        auxiliary2_prob_1x = outputs[3]
        auxiliary3_prob_1x = outputs[4]

        # dice loss
        self.main_weight_loss = softmax_loss_function(logits, labels, dst_weight)

        self.auxiliary1_weight_loss = softmax_loss_function(auxiliary1_prob_1x, labels, dst_weight)
        self.auxiliary2_weight_loss = softmax_loss_function(auxiliary2_prob_1x, labels, dst_weight)
        self.auxiliary3_weight_loss = softmax_loss_function(auxiliary3_prob_1x, labels, dst_weight)
        self.total_weight_loss = \
            self.main_weight_loss + \
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3

        self.main_jacc_loss = jacc_loss_new(logits, labels)
        self.auxiliary1_jacc_loss = jacc_loss_new(auxiliary1_prob_1x, labels)
        self.auxiliary2_jacc_loss = jacc_loss_new(auxiliary2_prob_1x, labels)
        self.auxiliary3_jacc_loss = jacc_loss_new(auxiliary3_prob_1x, labels)
        self.total_jacc_loss = \
            self.main_jacc_loss + \
            self.auxiliary1_jacc_loss * 0.8 + \
            self.auxiliary2_jacc_loss * 0.4 + \
            self.auxiliary3_jacc_loss * 0.2

        if cfg.only_jacc_loss == True:
            self.total_loss = self.total_jacc_loss
        elif cfg.jacc_entropy_loss == True:
            self.total_loss = self.total_jacc_loss + self.total_weight_loss

            tf.summary.scalar("main_jacc_loss", self.main_jacc_loss)
            tf.summary.scalar("total_jacc_loss", self.total_jacc_loss)
        elif cfg.use_dst_weight == True or cfg.use_weight == True:
            self.total_loss = self.total_weight_loss
        else:
            self.total_loss = self.total_weight_loss

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(pred_labels, tf.cast(labels, dtype=tf.int64))
            #correct_pred = tf.reduce_sum(pred_labels) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('seg accuracy', accuracy)
        tf.summary.scalar('seg loss', self.total_loss )

        return self.total_loss, accuracy
