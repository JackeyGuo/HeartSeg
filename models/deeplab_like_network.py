import tensorflow as tf
import config.config as cfg
from core.layers import _create_conv_layer, _conv2d_atrous, _batch_norm, _weight_variable_msra, conv2d
from core.loss_func import entropy_loss_function,softmax_loss_function,jacc_loss,jacc_loss_new

class Deeplab(object):
    def __init__(self,is_training):
        self.is_training = is_training

        self.output_classes = cfg.classes
    def _create_conv_layer(self, _input, kernel_size=3, stride=1, atrous=1, output_feature=16, padding="SAME"):

        if atrous == 1:
            output = conv2d(_input, kernel_size, stride, output_feature, padding)
        else:
            output = _conv2d_atrous(_input, kernel_size, stride, atrous, output_feature, padding)
        # output = self._batch_norm(output)
        # output = tf.nn.relu(output)
        return output

    def _create_block(self,_input,output_feature=64,atrous_rate=1):
        input_dim = int(_input.get_shape()[-1])
        if not input_dim == output_feature:
            with tf.variable_scope("conv0"):
                _input = _create_conv_layer(_input,output_feature=output_feature, atrous=atrous_rate,is_training=self.is_training)
                _input = _batch_norm(_input,self.is_training)

        with tf.variable_scope("conv1"):
            output = _create_conv_layer(_input,1,output_feature=output_feature // 4)
            output = _batch_norm(output,self.is_training)
        with tf.variable_scope("conv2"):
            output = _create_conv_layer(output,output_feature=output_feature // 4, atrous=atrous_rate,is_training=self.is_training)
            output = _batch_norm(output,self.is_training)
        with tf.variable_scope("conv3"):
            output = _create_conv_layer(output,kernel_size=1,output_feature=output_feature,is_training=self.is_training)
            output = _batch_norm(output,self.is_training)
        output = tf.add(_input,output)
        return output

    def _create_aspp(self,_input,size=4):
        output_size = int(_input.get_shape()[-1])
        aspps = []
        for i in range(size):
            with tf.variable_scope("aspp{0}".format(i)):
                out = self._create_conv_layer(_input,atrous=2**i,output_feature=output_size // 2)
                aspps.append(out)

        output = tf.add_n(aspps)
        return output


    def inference_op(self, _input):
        """
        the inference_op.and all the network has the same begin and end.
        :param _input:
        :return:
        """
        img = _input
        img_shape = img.get_shape()
        #output = _batch_norm(_input)
        output = conv2d(_input, 5, 2, 16, "SAME") # change to 1

        with tf.variable_scope("block1a"):
            output = self._create_block(output,32)
            output = tf.nn.relu(output)

        with tf.variable_scope("block1b"):
            output = self._create_block(output,32)
            output = tf.nn.relu(output)

        with tf.variable_scope("block2a"):
            output = self._create_block(output,64)
            output = tf.nn.relu(output)

        with tf.variable_scope("block2b"):
            output = self._create_block(output,64)
            output = tf.nn.relu(output)

        with tf.variable_scope("block2c"):
            output = self._create_block(output,64,2)
            output = tf.nn.relu(output)

        with tf.variable_scope("block3a"):
            output = self._create_block(output,128,4)
            output = tf.nn.relu(output)

        with tf.variable_scope("block3b"):
            output = self._create_block(output,128,4)
            output = tf.nn.relu(output)

        with tf.variable_scope("aspp"):
            output = self._create_aspp(output)

        in_channel = int(output.get_shape()[-1])
        kernel = _weight_variable_msra([3, 3, 8, in_channel], "upconv_kernel")
        output_shape = [int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), 8]
        output = tf.nn.conv2d_transpose(output, kernel, output_shape, [1, 2, 2, 1])
        output = tf.concat([img, output], -1)
        with tf.variable_scope("conv_bf1"):
            _output = self._create_conv_layer(output)
            output = tf.concat([_output, output], -1)
        # with tf.variable_scope("conv_bf2"):
        #     _output = self._create_conv_layer(output)
        #     output = tf.concat([_output, output], -1)
        with tf.variable_scope("fc_layer"):
            output = self._create_conv_layer(output, 1, output_feature=self.output_classes)
        with tf.variable_scope('prediction'):
            softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=-1, name='predicted_label')
        return output,predicted_label

    def loss_op(self, outputs, labels):
        logits = outputs[0]
        pred_labels = outputs[1]
        #print(logits,pred_labels)
        with tf.name_scope("weighted_cross_entropy"):
            #loss = jacc_loss_new(logits, labels)
            #labels = tf.one_hot(labels, depth=2)
            #labels = tf.cast(labels, dtype=tf.int64)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        with tf.name_scope("accuracy"):
            #correct_pred = tf.reduce_sum(logits[1]) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            #correct_pred = tf.equal(pred_labels, tf.cast(labels, dtype=tf.int64))
            correct_pred = tf.equal(pred_labels, tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar("total_loss", loss)

        return loss,accuracy
    '''def loss_op(self, logits, labels):
        # TODO:remove receptive_field! or put it into config file.
        receptive_field = 0
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

        logits_slice = tf.squeeze(logits_slice)
        labels_slice = tf.squeeze(labels_slice)
        labels_slice = tf.cast(labels_slice, dtype=tf.float32)

        weight_loss = cfg.weight_loss
        pw = 1
        if weight_loss == 0:
            # change it to calc weight.
            pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
        else:
            pw = weight_loss

        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=labels_slice, logits=logits_slice, pos_weight=pw))
        tf.summary.scalar("total_loss", loss)

        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.sigmoid(logits_slice) + 0.5, tf.int32), tf.cast(labels_slice, dtype=tf.int32)),
                    tf.float32), name="accuracy")
        tf.summary.scalar("accuracy", acc)
        return loss, acc'''


