from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from Dataset import save_images
import functools
from tfLib.ops import *
from tfLib.ops import instance_norm as IN
from tfLib.loss import *
from tfLib.advloss import *
import os

class Gaze_GAN(object):

    # build model
    def __init__(self, dataset, opt):

        self.dataset = dataset
        self.opt = opt
        # placeholder
        self.x_left_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.x_right_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.x = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.xm = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.output_nc])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self.xc = self.x * (1 - self.xm)  #corrputed images
        self.xl_left, self.xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)

        self.xl_left_fp = self.encode(self.xl_left)
        self.xl_right_fp = self.encode(self.xl_right)

        self.yo = self.G(self.xc, self.xm, self.xl_left_fp, self.xl_right_fp, use_sp=False)
        self.yl_left, self.yl_right = self.crop_resize(self.yo, self.x_left_p, self.x_right_p)

        self.yl_left_fp = self.encode(self.yl_left)
        self.yl_right_fp = self.encode(self.yl_right)

        self.y = self.xc + self.yo * self.xm

        if self.opt.is_ss:

            self.d_logits, self.d_logits_left, self.d_logits_right \
                = self.D(self.x, self.xl_left, self.xl_right, self.xl_left_fp, self.xl_right_fp)
            self.g_logits, self.g_logits_left, self.g_logits_right \
                = self.D(self.y, self.yl_left, self.yl_right, self.yl_left_fp, self.yl_right_fp)
            self.r_cls_loss = SSCE(labels=tf.zeros(shape=[self.opt.batch_size], dtype=tf.int32), logits=self.d_logits_left) + \
                        SSCE(labels=tf.ones(shape=[self.opt.batch_size], dtype=tf.int32), logits=self.d_logits_right)
            self.f_cls_loss = SSCE(labels=tf.zeros(shape=[self.opt.batch_size], dtype=tf.int32), logits=self.g_logits_left) + \
                                   SSCE(labels=tf.ones(shape=[self.opt.batch_size], dtype=tf.int32), logits=self.g_logits_right)

        else:
            self.d_logits = self.D(self.x, self.xl_left, self.xl_right,
                                                        self.xl_left_fp, self.xl_right_fp)
            self.g_logits = self.D(self.y, self.yl_left, self.yl_right,
                                                        self.yl_left_fp, self.yl_right_fp)

        d_loss_fun, g_loss_fun = get_adversarial_loss(self.opt.loss_type)
        self.d_gan_loss = d_loss_fun(self.d_logits, self.g_logits)
        self.g_gan_loss = g_loss_fun(self.g_logits)

        self.percep_loss = L1(self.xl_left_fp, self.yl_left_fp) + L1(self.xl_right_fp, self.yl_right_fp)
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y - self.x),
                            axis=[1, 2, 3]) / (self.opt.crop_w * self.opt.crop_h * self.opt.output_nc))

        if self.opt.is_ss:
            self.D_loss = self.d_gan_loss + self.opt.lam_ss * self.r_cls_loss
            self.G_loss = self.g_gan_loss + self.opt.lam_r * self.recon_loss \
                          + self.opt.lam_p * self.percep_loss + self.opt.lam_ss * self.f_cls_loss
        else:
            self.D_loss = self.d_gan_loss
            self.G_loss = self.g_gan_loss + self.opt.lam_r * self.recon_loss + self.opt.lam_p * self.percep_loss

    def build_test_model(self):
        self.xc = self.x * (1 - self.xm)
        self.xl_left, self.xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)
        self.xl_left_fp = self.encode(self.xl_left)
        self.xl_right_fp = self.encode(self.xl_right)
        self.yo = self.G(self.xc, self.xm, self.xl_left_fp, self.xl_right_fp, use_sp=False)
        self.y = self.xc + self.yo * self.xm


    def crop_resize(self, input, boxes_left, boxes_right):

        shape = [int(item) for item in input.shape.as_list()]
        return tf.image.crop_and_resize(input, boxes=boxes_left, box_ind=list(range(0, shape[0])),
                                        crop_size=[int(shape[-3] / 2), int(shape[-2] / 2)]), \
                tf.image.crop_and_resize(input, boxes=boxes_right, box_ind=list(range(0, shape[0])),
                                    crop_size=[int(shape[-3] / 2), int(shape[-2] / 2)])

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            print('Load checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
            else:
                print('Do not exists any checkpoint,Load Failed!')
                exit()

            _,_,_, testbatch, testmask = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 1000 / self.opt.batch_size
            for j in range(int(batch_num)):
                real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)
                f_d = {self.x: real_test_batch,
                       self.xm: batch_masks,
                       self.x_left_p: batch_left_eye_pos,
                       self.x_right_p: batch_right_eye_pos}
                
                output = sess.run([self.x, self.y], feed_dict=f_d)
                output_concat = self.Transpose(np.array([output[0], output[1]]))
                save_images(output_concat, '{}/{:02d}.jpg'.format(self.opt.test_sample_dir, j))

            coord.request_stop()
            coord.join(threads)

    def train(self):

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'D' in var.name]
        self.g_vars = [var for var in self.t_vars if 'G' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]
        assert len(self.t_vars) == len(self.d_vars + self.g_vars + self.e_vars)

        self.saver = tf.train.Saver()
        self.p_saver = tf.train.Saver(self.e_vars)

        opti_D = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2).\
                                        minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2).\
                                        minimize(loss=self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            start_step = 0
            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("")
                # try:
                #     #self.p_saver.restore(sess, os.path.join(self.opt.pretrain_path,
                #     #                                           'model_{:06d}.ckpt'.format(100000)))
                # except:
                #     print(" Self-Guided Model path may not be correct")

            #summary_op = tf.summary.merge_all()
            #summary_writer = tf.summary.FileWriter(self.opt.log_dir, sess.graph)
            step = start_step
            lr_decay = 1

            print("Start read dataset")

            image_path, train_images, train_eye_pos, test_images, test_eye_pos = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            real_test_batch, real_test_pos = sess.run([test_images, test_eye_pos])

            while step <= self.opt.niter:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.opt.niter - step) / float(self.opt.niter - 20000)

                real_batch_image_path, x_data, x_p_data = sess.run([image_path, train_images, train_eye_pos])
                xm_data, x_left_p_data, x_right_p_data = self.get_Mask_and_pos(x_p_data)

                f_d = {self.x: x_data,
                       self.xm: xm_data,
                       self.x_left_p: x_left_p_data,
                       self.x_right_p: x_right_p_data,
                       self.lr_decay: lr_decay}

                # optimize D
                sess.run(opti_D, feed_dict=f_d)
                # optimize G
                sess.run(opti_G, feed_dict=f_d)
                #summary_str = sess.run(summary_op, feed_dict=f_d)
                #summary_writer.add_summary(summary_str, step)
                if step % 500 == 0:

                    if self.opt.is_ss:
                        output_loss = sess.run([self.D_loss, self.G_loss, self.opt.lam_r * self.recon_loss,
                                                self.opt.lam_p * self.percep_loss, self.r_cls_loss, self.f_cls_loss], feed_dict=f_d)
                        print("step %d D_loss=%.8f, G_loss=%.4f, Recon_loss=%.4f, Percep_loss=%.4f, "
                              "Real_class_loss=%.4f, Fake_class_loss=%.4f, lr_decay=%.4f" %
                              (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], output_loss[4], output_loss[5], lr_decay))
                    else:
                        output_loss = sess.run([self.D_loss, self.G_loss, self.opt.lam_r * self.recon_loss,
                                                self.opt.lam_p * self.percep_loss], feed_dict=f_d)
                        print("step %d D_loss=%.8f, G_loss=%.4f, Recon_loss=%.4f, Percep_loss=%.4f, lr_decay=%.4f" %
                                        (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], lr_decay))

                if np.mod(step, 2000) == 0:

                    train_output_img = sess.run([self.xl_left, self.xl_right, self.xc,
                                                 self.yo, self.y, self.yl_left,
                                                 self.yl_right], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)
                    #for test
                    f_d = {self.x: real_test_batch, self.xm: batch_masks,
                           self.x_left_p: batch_left_eye_pos, self.x_right_p: batch_right_eye_pos,
                           self.lr_decay: lr_decay}

                    test_output_img = sess.run([self.xc, self.yo, self.y], feed_dict=f_d)
                    output_concat = self.Transpose(np.array([x_data, train_output_img[2], train_output_img[3],
                                                train_output_img[4]]))
                    local_output_concat = self.Transpose(np.array([train_output_img[0],
                                            train_output_img[1], train_output_img[5], train_output_img[6]]))
                    test_output_concat = self.Transpose(np.array([real_test_batch, test_output_img[0],
                                                                  test_output_img[2], test_output_img[1]]))
                    save_images(local_output_concat,
                                            '{}/{:02d}_local_output.jpg'.format(self.opt.sample_dir, step))
                    save_images(output_concat, '{}/{:02d}_output.jpg'.format(self.opt.sample_dir, step))
                    save_images(test_output_concat, '{}/{:02d}_test_output.jpg'.format(self.opt.sample_dir, step))

                if np.mod(step, 20000) == 0:
                    self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
            #summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def Transpose(self, list):

        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list

    def D(self, x, xl_left, xl_right, fp_left, fp_right):

        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("D", reuse=tf.AUTO_REUSE):

            xg_fp = self.global_d(x)
            if self.opt.is_ss:
                x1_left_fp, cls_left = self.local_d(xl_left)
                xl_right_fp, cls_right = self.local_d(xl_right)
                xl_fp = tf.concat([x1_left_fp, xl_right_fp], axis=-1)
            else:
                xl_fp = self.local_d(tf.concat([xl_left, xl_right], axis=-1))

            # Concatenation
            ful = tf.concat([xg_fp, xl_fp, fp_left, fp_right], axis=1)
            ful = tf.nn.relu(fc(ful, output_size=512, scope='fc1'))
            logits = fc(ful, output_size=1, scope='fc2')

            if self.opt.is_ss:
                return logits, cls_left, cls_right
            else:
                return logits

    def local_d(self, x):
        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d1", reuse=tf.AUTO_REUSE):
            for i in range(self.opt.n_layers_d):
                output_dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 256)
                x = lrelu(conv2d_base(x, output_dim=output_dim, scope='d{}'.format(i)))
            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            logits = fc(x, output_size=2, scope='logits')
            fp = fc(x, output_size=output_dim, scope='fp')
            if self.opt.is_ss:
                return logits, fp
            else:
                return logits

    def global_d(self, x):

        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d2", reuse=tf.AUTO_REUSE):

            # Global Discriminator Dg
            for i in range(self.opt.n_layers_d):
                dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 256)
                x = lrelu(conv2d_base(x, output_dim=dim, scope='d{}'.format(i)))

            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            fp = fc(x, output_size=dim, scope='fp')

            return fp

    def G(self, input_x, img_mask, fp_left, fp_right, use_sp=False):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        conv2d_final = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp, output_dim=self.opt.output_nc)
        with tf.variable_scope("G", reuse=tf.AUTO_REUSE):

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            x = lrelu(IN(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i+1), 256)
                x = lrelu(IN(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)),scope='IN{}'.format(i)))
                u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')
            bottleneck = tf.concat([bottleneck, fp_left, fp_right], axis=1)

            h, w = x.shape.as_list()[-3], x.shape.as_list()[-2]
            de_x = lrelu(fc(bottleneck, output_size=256*h*w, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.opt.batch_size, h, w, 256])

            ngf = c_dim
            for i in range(self.opt.n_layers_g):
                c_dim = np.maximum(int(ngf / np.power(2, i)), 16)
                de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i + 1)]], axis=3)
                de_x = tf.nn.relu(instance_norm(de_conv(de_x,
                            output_shape=[self.opt.batch_size, h*pow(2, i+1), w*pow(2, i+1), c_dim], use_sp=use_sp,
                            scope='deconv{}'.format(i)), scope='IN_{}'.format(i)))
            de_x = conv2d_final(de_x, scope='output_conv')

            return tf.nn.tanh(de_x)

    def encode(self, x):
        conv2d_first = functools.partial(conv2d, kernel=7, stride=1)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2)
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):

            nef = self.opt.nef
            x = tf.nn.relu(IN(conv2d_first(x, output_dim=nef, scope='e_c1'), scope='e_in1'))
            for i in range(self.opt.n_layers_e):
                x = tf.nn.relu(IN(conv2d_base(x, output_dim=min(nef * pow(2, i+1), 128), scope='e_c{}'.format(i+2)),
                                  scope='e_in{}'.format(i+2)))
            bottleneck = tf.reshape(x, [self.opt.batch_size, -1])
            content = fully_connect(bottleneck, output_size=128, scope='e_ful1')

            return content

    def get_Mask_and_pos(self, eye_pos, flag=0):

        eye_pos = eye_pos
        batch_mask = []
        batch_left_eye_pos = []
        batch_right_eye_pos = []
        for i in range(self.opt.batch_size):

            current_eye_pos = eye_pos[i]
            left_eye_pos = []
            right_eye_pos = []

            if flag == 0:

                mask = np.zeros(shape=[self.opt.img_size, self.opt.img_size, self.opt.output_nc])
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
                left_eye_pos.append(float(l1_1)/self.opt.img_size)
                left_eye_pos.append(float(l1_2)/self.opt.img_size)
                left_eye_pos.append(float(u1_1)/self.opt.img_size)
                left_eye_pos.append(float(u1_2)/self.opt.img_size)

                scale = current_eye_pos[3] - 15
                down_scale = current_eye_pos[3] + 15
                l2_1 = int(scale)
                u2_1 = int(down_scale)

                scale = current_eye_pos[2] - 25
                down_scale = current_eye_pos[2] + 25
                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

                right_eye_pos.append(float(l2_1) / self.opt.img_size)
                right_eye_pos.append(float(l2_2) / self.opt.img_size)
                right_eye_pos.append(float(u2_1) / self.opt.img_size)
                right_eye_pos.append(float(u2_2) / self.opt.img_size)

            batch_mask.append(mask)
            batch_left_eye_pos.append(left_eye_pos)
            batch_right_eye_pos.append(right_eye_pos)

        return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)




