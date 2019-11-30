import tensorflow as tf
from config import config as cfg
from core.loss_func import entropy_loss_function,softmax_loss_function,jacc_loss,jacc_loss_new,focal_loss_with_weight
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

class DenseDilatedASPP(object):
    def __init__(self,is_training):
        self.block_size = cfg.block_size
        self.block_count = cfg.block_count
        self.use_bc = cfg.use_bc
        self.block_output = []
        self.is_training = is_training
        self.output_classes = cfg.classes
    @staticmethod
    def _weight_variable_msra(shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def _batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, 0.99, scale=True,updates_collections=None)
        return output

    def _conv2d(self, _input, kernel_size, stride, output_feature, padding="SAME"):
        in_features = int(_input.get_shape()[-1])
        kernel = self._weight_variable_msra(
            [kernel_size, kernel_size, in_features, output_feature],
            name='kernel')
        strides = [1, stride, stride, 1]
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def _conv2d_atrous(self, _input, kernel_size=3, stride=1, atrous=1, output_feature=8, padding="SAME"):
        in_features = int(_input.get_shape()[-1])
        kernel = self._weight_variable_msra(
            [kernel_size, kernel_size, in_features, output_feature],
            name='kernel')
        strides = [stride, stride]
        atrous_size = [atrous, atrous]
        return tf.nn.convolution(_input, kernel, padding, strides, atrous_size)

    def _create_conv_layer(self, _input, kernel_size=3, stride=1, atrous=1, output_feature=16, padding="SAME"):
        _input = self._batch_norm(_input)
        _input = tf.nn.relu(_input)
        if atrous == 1:
            output = self._conv2d(_input, kernel_size, stride, output_feature, padding)
        else:
            output = self._conv2d_atrous(_input, kernel_size, stride, atrous, output_feature, padding)
        return output

    def _create_bc_layer(self, _input, output_feature):
        return self._create_conv_layer(_input, 1, output_feature=output_feature)

    def _create_pyramid(self, _input, magic_size):
        layers = []
        for i in range(magic_size):
            with tf.variable_scope("pyramid{0}".format(i)):
                output = self._create_conv_layer(_input, atrous=2 ** i)
                layers.append(output)
        output = tf.concat(layers, -1)
        return output

    def _create_block(self, _input, idx):
        """
        :param _input:
        :param idx:int, 1 block 1 idx.
        :return:
        """
        with tf.variable_scope("block{0}".format(idx)):
            layers = []
            # for i in range(self.block_size):
            #     if i == self.block_size - 1 and idx > 0:
            #         _input = self._create_pyramid(_input, idx + 1)
            #         _input = tf.concat([layers[-1], _input], -1)
            #     else:
            #         with tf.variable_scope("conv{0}".format(i)):
            #             _input = self._create_conv_layer(_input)
            #             if len(layers) > 0:
            #                 _input = tf.concat([layers[-1], _input], -1)
            #                 layers.append(_input)
            #             else:
            #                 layers.append(_input)
            for i in range(self.block_size+1):
                if i == self.block_size:
                        _input = self._create_pyramid(_input, idx + 1)
                        _input = tf.concat([layers[-1], _input], -1)
                else:
                    with tf.variable_scope("conv{0}".format(i)):
                        _input = self._create_conv_layer(_input)
                        if len(layers) > 0:
                            _input = tf.concat([layers[-1], _input], -1)
                            layers.append(_input)
                        else:
                            layers.append(_input)

        if self.use_bc:
            with tf.variable_scope("bc_layer{0}".format(idx)):
                output = self._create_bc_layer(_input, 8)
        else:
            output = _input
        return output

    def inference_op(self, _input):
        img = _input
        img_shape = img.get_shape()
        # output = self._create_conv_layer(_input, 5, 2)  # change to 1
        #output = self._batch_norm(_input)
        #output = _input / 255
        output = self._conv2d(_input, 5, 2, 16, "SAME")
        self.block_output.append(output)
        for i in range(self.block_count):
            output = self._create_block(output, i)
            output = tf.concat([self.block_output[-1], output], -1)

        in_channel = int(output.get_shape()[-1])
        kernel = self._weight_variable_msra([3, 3, 8, in_channel], "upconv_kernel")
        output_shape = [int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), 8]
        output = tf.nn.conv2d_transpose(output, kernel, output_shape, [1, 2, 2, 1])
        output = tf.concat([img, output], -1)
        print('output', output)
        with tf.variable_scope("conv_bf1"):
            _output = self._create_conv_layer(output)
            output = tf.concat([_output, output], -1)
        print('output', output)
        # with tf.variable_scope("conv_bf2"):
        #     _output = self._create_conv_layer(output)
        #     output = tf.concat([_output, output], -1)
        with tf.variable_scope("fc_layer"):
            output = self._create_conv_layer(output, 1, output_feature=1)
        print('output', output)

        # with tf.variable_scope('prediction'):
        #     softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
        #     predicted_label = tf.argmax(input=softmax_prob, axis=-1, name='predicted_label')
        return output

    def loss_op(self, outputs, labels, dst_weight=None):
        # TODO:remove receptive_field! or put it into config file.
        '''receptive_field = 0
        half_receptive = receptive_field // 2

        data_shape = labels.get_shape()
        transed_shape = [int(data_shape[0]), int(data_shape[1]), int(data_shape[2]), int(data_shape[3])]
        # slice the logits and labels to the non-padding area.
        logits_slice = tf.slice(logits, [0, half_receptive, half_receptive, half_receptive, 0],
                                [1, transed_shape[1] - receptive_field, transed_shape[2] - receptive_field,
                                 transed_shape[3] - receptive_field, 1])

        labels_slice = tf.slice(labels, [0, half_receptive, half_receptive, half_receptive],
                                [1, transed_shape[1] - receptive_field, transed_shape[2] - receptive_field,
                                 transed_shape[3] - receptive_field])
        print(logits_slice)
        logits_slice = tf.squeeze(logits_slice)
        labels_slice = tf.squeeze(labels_slice)
        labels_slice = tf.cast(labels_slice, dtype=tf.float32)

        weight_loss = cfg.weight_loss
        pw = 1
        if weight_loss == 0:
            pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
            # change it to calc weight.
        else:
            pw = weight_loss'''

        #loss = entropy_loss_function(logits, labels, dst_weight)

        logits = tf.squeeze(outputs)
        labels = tf.cast(labels, dtype=tf.float32)
        #pred_labels = outputs[1]

        weight_loss = cfg.weight_loss
        use_dst_loss = cfg.use_dst_loss
        focal_loss = cfg.focal_loss
        pw = 1
        if weight_loss == 0:
            #pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
            if use_dst_loss == 1 and dst_weight is not None:
                #print(dst_weight)
                weight = ops.convert_to_tensor(dst_weight)
                pw = weight
                # change it to calc weight.
        else:
            pw = weight_loss
        with tf.name_scope("weighted_cross_entropy"):
            # if cfg.use_jacc_loss == True:
            #     self.total_loss = jacc_loss_new(logits, labels)
            # else:
            #     self.total_loss = softmax_loss_function(logits, labels, dst_weight)
            if focal_loss == 1:
                focal_loss_r = cfg.focal_loss_r
                focal_loss_a = cfg.focal_loss_a
                # tf.abs(logits_slice - 0.5) + 0.5
                self.total_loss = tf.reduce_mean(focal_loss_with_weight(targets=labels,
                                                                        logits=logits,pos_weight=pw,r=focal_loss_r,a=focal_loss_a))
            else:
                self.total_loss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=pw))

        with tf.name_scope("accuracy"):
            # correct_pred = tf.equal(tf.cast(pred_labels,dtype=tf.int64), tf.cast(labels, dtype=tf.int64))
            # # correct_pred = tf.reduce_sum(pred_labels) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.sigmoid(logits) + 0.5, tf.int32), tf.cast(labels, dtype=tf.int32)),
                    tf.float32), name="accuracy")

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('total loss', self.total_loss )

        return self.total_loss, accuracy

