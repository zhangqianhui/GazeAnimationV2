# -*- coding: UTF-8 -*
import numpy as np
import tensorflow as tf
from Dataset import save_images
from ops import conv2d, lrelu, instance_norm, de_conv, fully_connect
import scipy
import tarfile, sys
from six.moves import urllib
import os, math

class Gaze_GAN(object):

    # build model
    def __init__(self, dataset, config):

        self.dataset = dataset

        # input hyper
        self.output_size = config.image_size
        self.channel = dataset.channel
        self.batch_size = config.batch_size
        self.pos_number = config.pos_number
        self.pretrain_model_index = config.pretrain_model_index
        self.pretrain_model_dir = config.pretrain_model_dir
        self.evaluation_path = config.evaluation_path
        self.MODEL_DIR = config.MODEL_DIR

        # output hyper
        self.sample_dir = config.sample_dir
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir
        self.is_supervised = config.is_supervised

        self.result_dir = config.result_dir
        self.testresult_dir = config.testresult_dir
        self.batch_num = dataset.test_num / self.batch_size

        # model hyper
        self.lam_percep = config.lam_percep
        self.lam_recon = config.lam_recon
        self.lam_ss = config.lam_ss
        self.loss_type = config.loss_type
        self.use_sp = config.use_sp
        self.log_vars = []

        # trainning hyper
        self.g_learning_rate = config.g_learning_rate
        self.d_learning_rate = config.d_learning_rate

        self.height = 30
        self.width = 50

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.lr_init = config.learning_rate_init
        self.max_iters = config.max_iters

        # placeholder
        self.input_left_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input_right_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.domain_label = tf.placeholder(tf.int32, [self.batch_size])

        self.list_all_members()

    def list_all_members(self):
        for name, value in vars(self).items():
            print('%s=%s' % (name, value))

    def build_model(self):

        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')
        self.incomplete_img = self.input * (1 - self.mask)
        self.local_input_left = self.crop_and_resize(self.input, self.input_left_labels)
        self.local_input_right = self.crop_and_resize(self.input, self.input_right_labels)

        self.angle_invar_left_real = self.encode(self.local_input_left, reuse=False)
        self.angle_invar_right_real = self.encode(self.local_input_right, reuse=True)

        self.recon_img = self.generator(self.incomplete_img, self.mask, self.angle_invar_left_real,
                                        self.angle_invar_right_real, reuse=False)

        self.local_recon_img_left = self.crop_and_resize(self.recon_img, self.input_left_labels)
        self.local_recon_img_right = self.crop_and_resize(self.recon_img, self.input_right_labels)

        self.angle_invar_left_recon = self.encode(self.local_recon_img_left, reuse=True)
        self.angle_invar_right_recon = self.encode(self.local_recon_img_right, reuse=True)

        self.new_recon_img = self.incomplete_img + self.recon_img * self.mask

        if self.is_supervised:

            self.D_real_gan_logits, self.real_logits_0, self.real_logits_1 \
                = self.discriminator(self.input, self.local_input_left, self.local_input_right,
                                                        self.angle_invar_left_real, self.angle_invar_right_real, reuse=False)
            self.D_fake_gan_logits, self.fake_logits_0, self.fake_logits_1 \
                = self.discriminator(self.new_recon_img, self.local_recon_img_left, self.local_recon_img_right,
                                                        self.angle_invar_left_real, self.angle_invar_right_real, reuse=True)
            self.real_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.one_hot(tf.zeros(shape=[self.batch_size], dtype=tf.int32), 2), logits=self.real_logits_0)) + \
                                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.one_hot(tf.ones(shape=[self.batch_size], dtype=tf.int32), 2), logits=self.real_logits_1))

            self.fake_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.one_hot(tf.zeros(shape=[self.batch_size], dtype=tf.int32), 2), logits=self.fake_logits_0)) + \
                                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.one_hot(tf.ones(shape=[self.batch_size], dtype=tf.int32), 2), logits=self.fake_logits_1))

        else:

            self.D_real_gan_logits = self.discriminator(self.input, self.local_input_left, self.local_input_right,
                                                        self.angle_invar_left_real, self.angle_invar_right_real, reuse=False)
            self.D_fake_gan_logits = self.discriminator(self.new_recon_img, self.local_recon_img_left, self.local_recon_img_right,
                                                        self.angle_invar_left_real, self.angle_invar_right_real, reuse=True)

        if self.loss_type == 0:
            self.d_gan_loss = self.loss_hinge_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_hinge_gen(self.D_fake_gan_logits)
        elif self.loss_type == 1:
            self.d_gan_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_gen(self.D_fake_gan_logits)
        elif self.loss_type == 2:
            self.d_gan_loss = self.d_lsgan_loss(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.g_lsgan_loss(self.D_fake_gan_logits)

        self.percep_loss = tf.reduce_mean(tf.abs(self.angle_invar_left_real - self.angle_invar_left_recon)) + \
                           tf.reduce_mean(tf.abs(self.angle_invar_right_real - self.angle_invar_right_recon))

        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.new_recon_img - self.input),
                                                              axis=[1, 2, 3])/ (self.width * self.height * self.channel))

        if self.is_supervised:
            self.D_loss = self.d_gan_loss + self.lam_ss * self.real_class_loss
            self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss \
                          + self.lam_percep * self.percep_loss + self.lam_ss * self.fake_class_loss
        else:
            self.D_loss = self.d_gan_loss
            self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss + self.lam_percep * self.percep_loss


        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.ed_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]

        self.saver = tf.train.Saver()
        self.pretrain_saver = tf.train.Saver(self.e_vars)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)


    def crop_and_resize(self,input, boxes):
       return tf.image.crop_and_resize(input, boxes=boxes,box_ind=range(0, self.batch_size),
                                       crop_size=[self.output_size / 2, self.output_size / 2])

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    def loss_hinge_dis(self, d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(self, d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    def d_lsgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(self, d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            print('Load checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                print ckpt.model_checkpoint_path
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
            else:
                print('Do not exists any checkpoint,Load Failed!')
                exit()

            _,_,_, testbatch, testmask = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 1000 / self.batch_size
            for j in range(batch_num):
                real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)
                f_d = {self.input: real_test_batch,
                       self.mask: batch_masks,
                       self.input_left_labels: batch_left_eye_pos,
                       self.input_right_labels: batch_right_eye_pos}

                for i in range(self.batch_size):

                    output = sess.run([self.input, self.new_recon_img], feed_dict=f_d)

                    save_images(np.reshape(output[0][i], newshape=(1, self.output_size, self.output_size, self.channel)),
                                [1, 1], '{}/{:02d}_{:2d}_input.jpg'.format(self.testresult_dir + "4", j, i))
                    save_images(np.reshape(output[1][i], newshape=(1, self.output_size, self.output_size, self.channel)),
                                [1, 1], '{}/{:02d}_{:2d}_recon.jpg'.format(self.testresult_dir + "3", j, i))
                    save_images(np.reshape(output[3][i], newshape=(1, self.output_size/2, self.output_size/2, self.channel)),
                                [1, 1], '{}/{:02d}_{:2d}_recon_local_right.jpg'.format(self.testresult_dir, j, i))

            coord.request_stop()
            coord.join(threads)

    def train(self):

        opti_D = tf.train.AdamOptimizer(self.d_learning_rate * self.lr_decay,beta1=self.beta1, beta2=self.beta2).\
                                        minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.g_learning_rate * self.lr_decay,beta1=self.beta1, beta2=self.beta2).\
                                        minimize(loss=self.G_loss, var_list=self.ed_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            start_step = 0
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                try:
                    self.pretrain_saver.restore(sess, os.path.join(self.pretrain_model_dir,
                                                               'model_{:06d}.ckpt'.format(self.pretrain_model_index)))
                except:
                    print(" Self-Guided Model path may not be correct")

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = start_step
            lr_decay = self.lr_init

            print("Start read dataset")

            image_path, train_images, train_eye_pos, test_images, test_eye_pos = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            real_test_batch, real_test_pos = sess.run([test_images, test_eye_pos])

            while step <= self.max_iters:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                real_batch_image_path, real_batch_image, real_eye_pos = sess.run([image_path, train_images, train_eye_pos])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)


                f_d = {self.input: real_batch_image,
                       self.mask: batch_masks,
                       self.input_left_labels: batch_left_eye_pos,
                       self.input_right_labels: batch_right_eye_pos,
                       self.lr_decay: lr_decay}

                # optimize D
                sess.run(opti_D, feed_dict=f_d)
                # optimize G
                sess.run(opti_G, feed_dict=f_d)

                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:

                    if self.is_supervised:

                        output_loss = sess.run([self.D_loss, self.G_loss, self.lam_recon * self.recon_loss,
                                                self.lam_percep * self.percep_loss, self.real_class_loss, self.fake_class_loss], feed_dict=f_d)
                        print("step %d D_loss=%.8f, G_loss=%.4f, Recon_loss=%.4f, Percep_loss=%.4f, "
                              "Real_class_loss=%.4f, Fake_class_loss=%.4f, lr_decay=%.4f" %
                              (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], output_loss[4], output_loss[5], lr_decay))

                    else:

                        output_loss = sess.run([self.D_loss, self.G_loss, self.lam_recon * self.recon_loss,
                                                self.lam_percep * self.percep_loss], feed_dict=f_d)

                        print("step %d D_loss=%.8f, G_loss=%.4f, Recon_loss=%.4f, Percep_loss=%.4f, lr_decay=%.4f" %
                                        (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], lr_decay))

                if np.mod(step, 2000) == 0:

                    train_output_img = sess.run([self.local_input_left, self.local_input_right, self.incomplete_img,
                                                 self.recon_img,self.new_recon_img, self.local_recon_img_left,
                                                 self.local_recon_img_right], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)

                    #for test
                    f_d = {self.input: real_test_batch, self.mask: batch_masks,
                           self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos,
                           self.lr_decay: lr_decay}

                    test_output_img = sess.run([self.incomplete_img, self.recon_img, self.new_recon_img], feed_dict=f_d)
                    output_concat = np.concatenate([real_batch_image, train_output_img[2], train_output_img[3],
                                                    train_output_img[4]], axis=0)
                    local_output_concat = np.concatenate([train_output_img[0], train_output_img[1], train_output_img[5],
                                                         train_output_img[6]], axis=0)
                    test_output_concat = np.concatenate([real_test_batch, test_output_img[0], test_output_img[2],
                                                         test_output_img[1]], axis=0)

                    save_images(local_output_concat, [local_output_concat.shape[0] / self.batch_size, self.batch_size],
                                            '{}/{:02d}_local_output.jpg'.format(self.sample_dir, step))
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_dir, step))
                    save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_test_output.jpg'.format(self.sample_dir, step))

                if np.mod(step, 20000) == 0:
                    self.Inception_score(sess, test_images, test_eye_pos, step)
                    self.FID_score(sess, test_images, test_eye_pos, step)
                    self.saver.save(sess, os.path.join(self.model_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.model_dir, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def discriminator(self, incom_x, local_x_left, local_x_right, guided_fp_left, guided_fp_right, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()
            # Global Discriminator Dg
            x = incom_x
            for i in range(6):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_global_{}'.format(i)))

            x = tf.reshape(x, shape=[self.batch_size, -1])
            ful_global = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_FC1')

            if self.is_supervised:

                with tf.variable_scope("local_d"):
                    # Local Discriminator Dl
                    x = local_x_left
                    for i in range(5):
                        output_dim = np.minimum(16 * np.power(2, i+1), 256)
                        x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_local_{}'.format(i)))
                    x = tf.reshape(x, shape=[self.batch_size, -1])
                    logits0 = fully_connect(x, output_size=2, use_sp=self.use_sp, scope='dis_class')
                    ful_local_left = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_FC2')

                with tf.variable_scope("local_d") as scope:
                    scope.reuse_variables()

                    x = local_x_right
                    for i in range(5):
                        output_dim = np.minimum(16 * np.power(2, i + 1), 256)
                        x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_local_{}'.format(i)))

                    x = tf.reshape(x, shape=[self.batch_size, -1])
                    logits1 = fully_connect(x, output_size=2, use_sp=self.use_sp, scope='dis_class')
                    ful_local_right = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_FC2')

                ful_local = tf.concat([ful_local_left, ful_local_right], axis=1)

            else:

                x = tf.concat([local_x_left, local_x_right], axis=3)
                for i in range(5):
                    output_dim = np.minimum(16 * np.power(2, i + 1), 256)
                    x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_local_{}'.format(i)))

                x = tf.reshape(x, shape=[self.batch_size, -1])
                ful_local = fully_connect(x, output_size=output_dim * 2, use_sp=self.use_sp, scope='dis_FC2')

            # Concatenation
            ful = tf.concat([ful_global, ful_local, guided_fp_left, guided_fp_right], axis=1)
            ful = tf.nn.relu(fully_connect(ful, output_size=512, use_sp=self.use_sp, scope='dis_FC3'))
            gan_logits = fully_connect(ful, output_size=1, use_sp=self.use_sp, scope='dis_FC4')

            if self.is_supervised:
                return gan_logits, logits0, logits1

            else:
                return gan_logits

    def generator(self, input_x, img_mask, guided_fp_left, guided_fp_right, use_sp=False, reuse=False):

        with tf.variable_scope("generator") as scope:

            if reuse == True:
                scope.reuse_variables()

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            for i in range(6):
                c_dim = np.minimum(16 * np.power(2, i), 256)
                if i == 0:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=7, k_h=7, d_w=1, d_h=1, use_sp=use_sp,
                                             name='conv_{}'.format(i)),scope='conv_IN_{}'.format(i)))
                else:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=use_sp,
                                             name='conv_{}'.format(i)),scope='conv_IN_{}'.format(i)))
                    if i < 5:
                        u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.batch_size, -1])
            bottleneck = fully_connect(bottleneck, output_size=256, use_sp=use_sp, scope='FC1')
            bottleneck = tf.concat([bottleneck, guided_fp_left, guided_fp_right], axis=1)

            de_x = tf.nn.relu(fully_connect(bottleneck, output_size=256*8*8, use_sp=use_sp, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.batch_size, 8, 8, 256])

            for i in range(5):
                c_dim = np.maximum(256 / np.power(2, i), 16)
                output_dim = 16 * np.power(2, i)
                de_x = tf.nn.relu(instance_norm(de_conv(de_x, output_shape=[self.batch_size, output_dim, output_dim, c_dim], use_sp=use_sp,
                                                            name='deconv_{}'.format(i)), scope='deconv_IN_{}'.format(i)))
                if i < 4:
                    de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i+1)]], axis=3)

            recon_img1 = conv2d(de_x, output_dim=3, k_w=7, k_h=7, d_h=1, d_w=1, use_sp=use_sp, name='output_conv')
            return tf.nn.tanh(recon_img1)

    def encode(self, x, reuse=False):

        with tf.variable_scope("encode") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x, output_dim=32, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=64, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))
            conv4 = tf.nn.relu(
                instance_norm(conv2d(conv3, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c4'), scope='e_in4'))

            bottleneck = tf.reshape(conv4, [self.batch_size, -1])
            content = fully_connect(bottleneck, output_size=128, scope='e_ful1')

            return content

    def get_Mask_and_pos(self, eye_pos, flag=0):

        eye_pos = eye_pos
        batch_mask = []
        batch_left_eye_pos = []
        batch_right_eye_pos = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            left_eye_pos = []
            right_eye_pos = []

            if flag == 0:

                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_eye_pos[1] - 15
                down_scale = current_eye_pos[1] + 15
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_eye_pos[0] - 25
                down_scale = current_eye_pos[0] + 25
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
                left_eye_pos.append(float(l1_1)/self.output_size)
                left_eye_pos.append(float(l1_2)/self.output_size)
                left_eye_pos.append(float(u1_1)/self.output_size)
                left_eye_pos.append(float(u1_2)/self.output_size)

                scale = current_eye_pos[3] - 15
                down_scale = current_eye_pos[3] + 15
                l2_1 = int(scale)
                u2_1 = int(down_scale)

                scale = current_eye_pos[2] - 25
                down_scale = current_eye_pos[2] + 25
                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

                right_eye_pos.append(float(l2_1) / self.output_size)
                right_eye_pos.append(float(l2_2) / self.output_size)
                right_eye_pos.append(float(u2_1) / self.output_size)
                right_eye_pos.append(float(u2_2) / self.output_size)

            batch_mask.append(mask)
            batch_left_eye_pos.append(left_eye_pos)
            batch_right_eye_pos.append(right_eye_pos)

        return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)

    #for columbia
    def get_eye_region(self, eye_pos, flag=0):

        batch_mask = []
        eye_pos = eye_pos
        # print eye_pos
        batch_left_eye_pos = []
        batch_right_eye_pos = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            left_eye_pos = []
            right_eye_pos = []
            # eye
            if flag == 0:

                # left eye, y
                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])

                scale = current_eye_pos[0] - self.height / 2  # current_eye_pos[3] / 2
                down_scale = current_eye_pos[0] + self.height / 2  # current_eye_pos[3] / 2
                l1_1 = int(scale)
                u1_1 = int(down_scale)

                # x
                scale = current_eye_pos[1] - self.width / 2  # current_eye_pos[2] / 2
                down_scale = current_eye_pos[1] + self.width / 2  # current_eye_pos[2] / 2
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0

                left_eye_pos.append(float(l1_1) / self.output_size)
                left_eye_pos.append(float(l1_2) / self.output_size)
                left_eye_pos.append(float(u1_1) / self.output_size)
                left_eye_pos.append(float(u1_2) / self.output_size)

                # right eye, y
                scale = current_eye_pos[2] - self.height / 2  # current_eye_pos[7] / 2
                down_scale = current_eye_pos[2] + self.height / 2  # current_eye_pos[7] / 2

                l2_1 = int(scale)
                u2_1 = int(down_scale)

                # x
                scale = current_eye_pos[3] - self.width / 2  # current_eye_pos[6] / 2
                down_scale = current_eye_pos[3] + self.width / 2  # current_eye_pos[6] / 2

                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

                right_eye_pos.append(float(l2_1) / self.output_size)
                right_eye_pos.append(float(l2_2) / self.output_size)
                right_eye_pos.append(float(u2_1) / self.output_size)
                right_eye_pos.append(float(u2_2) / self.output_size)

            batch_mask.append(mask)
            batch_left_eye_pos.append(left_eye_pos)
            batch_right_eye_pos.append(right_eye_pos)

        return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)


    def _init_inception(self):
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(self.MODEL_DIR, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(self.MODEL_DIR)
        with tf.gfile.FastGFile(os.path.join(
                self.MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='InputTensor')
    	    _ = tf.import_graph_def(graph_def, name='',
                            input_map={'ExpandDims:0':input_tensor})
		
        # Works with an arbitrary minibatch size.

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config, graph=tf.get_default_graph()) as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            ops = pool3.graph.get_operations()
            for op_idx, op in enumerate(ops):
                for o in op.outputs:
                    #print(o)
                    shape = o.get_shape()
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.set_shape(tf.TensorShape(new_shape))

            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3,[1,2]), w)

            self.softmax = tf.nn.softmax(logits)
            self.pool3 = pool3

    def get_data_feeded(self, sess, train_images, train_eye_pos):

        real_batch_image, real_eye_pos = sess.run([train_images, train_eye_pos])
        batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)

        f_d = {self.input: real_batch_image,
               self.mask: batch_masks,
               self.input_left_labels: batch_left_eye_pos,
               self.input_right_labels: batch_right_eye_pos}

        return f_d

    def transform_image(self, image):
        return (image + 1) * 127.5

    def Inception_score(self, sess, train_images, train_eye_pos, step):

        print("Computing Inception scores")
        bs = self.batch_size
        validate_numbers = 10000
        preds1 = []
        preds2 = []
        n_batches = int(math.ceil(float(validate_numbers) / float(bs)))
        for i in range(n_batches):

            if i % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            f_d = self.get_data_feeded(sess, train_images=train_images, train_eye_pos=train_eye_pos)
            validate_output_img = sess.run([self.local_input_left,
                                            self.local_input_right,
                                            self.local_recon_img_left,
                                            self.local_recon_img_right
                                            ], feed_dict=f_d)

            pred1 = sess.run(self.softmax, {'InputTensor:0': self.transform_image(validate_output_img[2])})
            preds1.append(pred1)
            pred2 = sess.run(self.softmax, {'InputTensor:0': self.transform_image(validate_output_img[3])})
            preds2.append(pred2)

        preds1 = np.concatenate(preds1, 0)
        scores1 = []
        for i in range(10):
            part = preds1[(i * preds1.shape[0] // 10):((i + 1) * preds1.shape[0] // 10), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores1.append(np.exp(kl))

        print("\n%d", step, "Inception Scores for Left Eyes", str(np.mean(scores1)) + "+-" + str(np.std(scores1)))

        preds2 = np.concatenate(preds2, 0)
        scores2 = []
        for i in range(10):
            part = preds2[(i * preds2.shape[0] // 10):((i + 1) * preds2.shape[0] // 10), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores2.append(np.exp(kl))
        print("\n%d", step, "Inception Scores for Right Eyes", str(np.mean(scores2)) + "+-" + str(np.std(scores2)))

        f = open('{}/incpetion_socre_log2.txt'.format(self.evaluation_path), 'a')  # append write
        f.writelines('{:04}:{}: {}\n'.format(step, str(np.mean(scores1)) + "+-" + str(np.std(scores1)),
                                             str(np.mean(scores2)) + "+-" + str(np.std(scores2))))
        f.close()

    def FID_score(self,sess, train_images, train_eye_pos, step):

        bs = self.batch_size
        validate_numbers = 10000
        nbatches = validate_numbers // bs
        n_used_imgs = nbatches * bs
        pred_arr1 = np.empty((n_used_imgs, 2048))
        pred_arr2 = np.empty((n_used_imgs, 2048))
        pred_arr_r = np.empty((n_used_imgs, 2048))
        pred_arr_r2 = np.empty((n_used_imgs, 2048))

        for i in range(nbatches):

            if i % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            start = i * bs
            end = start + bs

            f_d = self.get_data_feeded(sess, train_images=train_images, train_eye_pos=train_eye_pos)

            validate_output_img = sess.run([self.local_input_left,
                                            self.local_input_right,
                                            self.local_recon_img_left,
                                            self.local_recon_img_right
                                            ], feed_dict=f_d)

            pred = sess.run(self.pool3, {'InputTensor:0': self.transform_image(validate_output_img[2])})
            pred_arr1[start:end] = pred.reshape(bs, -1)
            pred = sess.run(self.pool3, {'InputTensor:0': self.transform_image(validate_output_img[3])})
            pred_arr2[start:end] = pred.reshape(bs, -1)
            # Get the activations for real sampels
            pred2 = sess.run(self.pool3, {'InputTensor:0': self.transform_image(validate_output_img[0])})
            pred_arr_r[start:end] = pred2.reshape(bs, -1)

            pred2 = sess.run(self.pool3, {'InputTensor:0': self.transform_image(validate_output_img[1])})
            pred_arr_r2[start:end] = pred2.reshape(bs, -1)

        g_mu1 = np.mean(pred_arr1, axis=0)
        g_sigma1 = np.cov(pred_arr1, rowvar=False)

        g_mu2 = np.mean(pred_arr2, axis=0)
        g_sigma2 = np.cov(pred_arr2, rowvar=False)

        g_mu_r = np.mean(pred_arr_r, axis=0)
        g_sigma_r = np.cov(pred_arr_r, rowvar=False)

        g_mu_r2 = np.mean(pred_arr_r2, axis=0)
        g_sigma_r2 = np.cov(pred_arr_r2, rowvar=False)

        m = np.square(g_mu1 - g_mu_r).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(g_sigma1, g_sigma_r), disp=False)  # EDIT: added
        dist1 = m + np.trace(g_sigma1 + g_sigma_r - 2 * s)

        print("FID1 for left: ", np.real(dist1))

        m = np.square(g_mu2 - g_mu_r2).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(g_sigma2, g_sigma_r2), disp=False)  # EDIT: added
        dist2 = m + np.trace(g_sigma2 + g_sigma_r2 - 2 * s)

        print("FID2 for right: ", np.real(dist2))

        f = open('{}/fid_socre_log2.txt'.format(self.evaluation_path), 'a')  # append write
        f.writelines('{:04}:{}: {}\n'.format(step, str(np.real(dist1)), str(np.real(dist2))))
        f.close()





