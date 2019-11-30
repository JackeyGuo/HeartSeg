import os
import random
import numpy as np
import tensorflow as tf
#from tools import scorer
from core import utils
from core import model_factory
import config.config as cfg

class Solver(object):
    """
    this class is the base processor.
    """
    def __init__(self,model_class,provider):
        """
        :param model_class:the Class of model.for pass the is_training.
        :param provider: the instance of the provider.
        """
        self.is_training = tf.Variable(tf.constant(True,tf.bool))
        self.model_class = model_class
        self.model_name = cfg.name
        self.provider = provider
        self.train_step = cfg.iter_step
        self.output_path = cfg.output_path

        self.save_interval = cfg.save_interval

        self.output_channels = 1

        #self.batch_size =1
        #self.output_size = 2

    def _load_model(self, saver, sess, model_path):

        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        #print(latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Checkpoint loaded")
        else:
            print("First time to train!")

    def _count_trainables(self):
        variables = tf.trainable_variables()
        #print(variables)
        counter = 0
        for i in variables:
            shape = i.get_shape()
            val = 4
            for j in shape:
                val *= int(j)

            counter += val

        print("with: {0} trainables".format(counter))

        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total number of trainable parameters: %d' % total_parameters)

    def _get_optimizer(self,loss,opt_name="adam"):
        decay_steps = cfg.lr_decay_step
        decay_rate = cfg.lr_decay_rate
        learning_rate = cfg.learn_rate

        self.global_step = tf.get_variable(
            'global_step', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                        self.global_step,
                                        decay_steps,
                                        decay_rate,
                                        staircase=True,
                                        name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(loss, global_step=self.global_step)

        return self.optimizer

    def train(self,fold_num):
        if cfg.name == 'denseDilatedASPP':
            cfg.use_dst_loss = cfg.use_dst_loss
        else:
            cfg.use_dst_loss = 0

        if cfg.use_dst_loss == 1:
            train_holder, seg_holder, dst_holder = self.provider.get_train_holder(batch_size=cfg.batch_size)
        else:
            train_holder, seg_holder = self.provider.get_train_holder(batch_size=cfg.batch_size)

        model = self.model_class(self.is_training)
        inference_op = model.inference_op(train_holder)

        if cfg.use_dst_loss == 1:
            loss_op, acc_op = model.loss_op(inference_op, seg_holder, dst_holder)
        else:
            loss_op, acc_op = model.loss_op(inference_op, seg_holder)
        train_op = self._get_optimizer(loss_op)

        merged = tf.summary.merge_all()
        self._count_trainables()
        log_output_path = os.path.join(self.output_path,"log")
        if not os.path.exists(log_output_path):
            os.makedirs(log_output_path)

        model_output_path = os.path.join(self.output_path,"model")
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        loss_txt_path = os.path.join(self.output_path, "loss")
        if not os.path.exists(loss_txt_path):
            os.makedirs(loss_txt_path)

        train_writer = tf.summary.FileWriter(os.path.join(log_output_path, "train"))
        test_writer = tf.summary.FileWriter(os.path.join(log_output_path, "val"))

        line_buffer = 1
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config = config
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            with open(file=loss_txt_path + '/loss_' + cfg.name + str(fold_num) + '.txt', mode='w', buffering=line_buffer) as loss_log:
                for step in range(self.train_step):
                    if cfg.use_dst_loss == 1:
                        image, label, weight = self.provider.get_train_value(batch_size=cfg.batch_size,
                                                                     with_weight=True)
                        image_val, label_val, weight_val = self.provider.get_val_value(batch_size=cfg.batch_size,
                                                                           with_weight=True)
                        train_merge, train_loss, _, train_acc = sess.run(
                            [merged, loss_op, train_op, acc_op],
                            feed_dict={train_holder: image, seg_holder: label, dst_holder: weight})
                        valid_merge, val_loss, val_acc = sess.run(
                            [merged, loss_op, acc_op],
                            feed_dict={train_holder: image_val, seg_holder: label_val, dst_holder:weight_val, self.is_training: False})
                    else:
                        image, label = self.provider.get_train_value(batch_size=cfg.batch_size)
                        image_val, label_val = self.provider.get_val_value(batch_size=cfg.batch_size)

                        train_merge, train_loss, _, train_acc = sess.run(
                            [merged, loss_op, train_op, acc_op],
                            feed_dict={train_holder: image, seg_holder: label})
                        valid_merge, val_loss, val_acc = sess.run(
                            [merged, loss_op, acc_op],
                            feed_dict={train_holder: image_val, seg_holder: label_val, self.is_training: False})

                    if np.mod(step + 1, self.save_interval) == 0:
                        saver.save(sess, os.path.join(self.output_path,"model/model_saved_%d"%fold_num))
                    output_format = "train loss: %f, valid loss: %f, train accuracy: %f, val accuracy: %f, step: %d" % \
                                    (train_loss, val_loss, train_acc, val_acc, step)
                    print(output_format)
                    train_writer.add_summary(train_merge, step)
                    test_writer.add_summary(valid_merge, step)

                    if step % 5 == 0:
                        loss_log.write(output_format + '\n')

                train_writer.close()
                test_writer.close()

    def predict(self,fold_num):
        tf.reset_default_graph()

        is_training = tf.Variable(tf.constant(False))
        test_holder = self.provider.get_test_holder()

        model = self.model_class(is_training)
        if cfg.name == 'vnet_2d' or cfg.name == 'unet_2d' or cfg.name == 'cnn_v2' \
                or cfg.name == 'deeplab':
            _,predict_label = model.inference_op(test_holder)
        elif cfg.name == 'denseDilatedASPP':
            output_op = model.inference_op(test_holder)
            predict_label = tf.sigmoid(output_op)
        else:
            _,predict_label,_,_,_ = model.inference_op(test_holder)

        with tf.Session() as sess:
            # TODO: load pre-trained model
            # TODO: load checkpoint
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.output_path, "model/model_saved_%d"%fold_num))
            self.provider.init_test()
            while True:
                test_lst = self.provider.get_test_value()

                if test_lst is None:
                    break
                output_lst = []
                for list in test_lst:
                    output = sess.run(predict_label, feed_dict={test_holder: list})
                    output_lst.append(output)
                self.provider.write_test(output_lst)

def main():

    if cfg.gpu:
        gpu = cfg.gpu
    else:
        gpu = ''

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    data_npy_path = os.path.join(cfg.output_path, "data")
    if not os.path.exists(data_npy_path):
        os.makedirs(data_npy_path)

    model = model_factory.modelMap[cfg.name]
    provider = utils.DataProvider()

    #patient_list = provider.getPatientName()
    #print(len(patient_list))
    #val_size = len(patient_list) // cfg.fold_num

    #pat_iter = 0
    #random.shuffle(patient_list)
    count = 1
    if cfg.one_by_one == False:
        patient_list = provider.getPatientName()
        val_size = len(patient_list) // cfg.fold_num

        pat_iter = 0
        random.shuffle(patient_list)
        # write patient_list to npy
        np.save(data_npy_path + '/data.npy', patient_list)
    else:
        data_path = os.path.join(cfg.output_path, "data/data.npy")
        data = np.load(data_path)
        print('load data successfully!!!!!!!!!!!')
        patient_list = []
        for i in range(len(data)):
            patient_list.append(data[i])
        val_size = len(patient_list) // cfg.fold_num
        # fold_iter = cfg.fold_iter
        pat_iter = (cfg.begin_fold_iter - 1) * val_size
        count = cfg.begin_fold_iter
        end_iter = (cfg.end_fold_iter) * val_size
    while True:
        if cfg.one_by_one == False and pat_iter == len(patient_list):
            break
        elif cfg.one_by_one == True and pat_iter == len(patient_list):
            break
        else:
            if count == cfg.fold_num:
                # split train val test data, train : 391, val : 97
                val_set = patient_list[pat_iter:]
                train_set = patient_list[:pat_iter]
                pat_iter = len(patient_list)
            else:
                # split train val test data, train : 391, val : 97
                val_set = patient_list[pat_iter:pat_iter + val_size]
                train_set = patient_list[:pat_iter] + patient_list[pat_iter + val_size:]
                pat_iter += val_size

            provider.setTrainVal(train_set, val_set)
            tf.reset_default_graph()

            processor = Solver(model, provider)

            processor.train(count)
            processor.predict(count)
            count += 1

    # caculate score of dice
    '''truth_path = cfg.data_path
    #output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/score" % (cfg.name)
    output_path = os.path.join(cfg.output_path, "score")
    #predict_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s_sqrt/predict" % (cfg.name)
    predict_path = os.path.join(cfg.output_path, "predict")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tpo = scorer.create_t_p_o_list(truth_path, predict_path, output_path)
    scorer.score(tpo)'''

if __name__ == "__main__":
    main()