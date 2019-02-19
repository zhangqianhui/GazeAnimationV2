import tensorflow as tf
from ops import conv2d, lrelu, instance_norm, de_conv, fully_connect
from Dataset2 import save_images
import os
import numpy as np

class Inpainting_GAN(object):

    # build model
    def __init__(self, data_ob, config):

        self.batch_size = config.batch_size
        self.max_iters = config.max_iters
        self.read_model_path = config.read_model_path
        self.write_model_path = config.write_model_path
        self.data_ob = data_ob
        self.sample_path = config.sample_path
        self.test_sample_path = config.test_sample_path
        self.log_dir = config.log_dir
        self.g_learning_rate = config.g_learning_rate
        self.d_learning_rate = config.d_learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.lam_recon = config.lam_recon
        self.lam_fp = config.lam_fp
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.output_size = config.image_size
        self.use_sp = config.use_sp
        self.pos_number = config.pos_number
        self.loss_type = config.loss_type
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.input_left_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input_right_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input_masks = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])

        self.domain_label = tf.placeholder(tf.int32, [self.batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self.incomplete_img = self.input * (1 - self.input_masks)
        self.local_input_left = tf.image.crop_and_resize(self.input, boxes=self.input_left_labels,
                                                         box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])

        self.local_input_right = tf.image.crop_and_resize(self.input, boxes=self.input_right_labels,
                                                         box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])

        self.guided_fp_x_left = self.encode2(self.local_input_left, reuse=False)
        self.guided_fp_x_right = self.encode2(self.local_input_right, reuse=True)

        self.x_tilde = self.encode_decode(self.incomplete_img, self.input_masks, self.guided_fp_x_left, self.guided_fp_x_right, reuse=False)

        self.local_x_tilde_left = tf.image.crop_and_resize(self.x_tilde, boxes=self.input_left_labels,
                                                         box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])

        self.local_x_tilde_right = tf.image.crop_and_resize(self.x_tilde, boxes=self.input_right_labels,
                                                         box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])

        self.guided_fp_x_tilde_left = self.encode2(self.local_x_tilde_left, reuse=True)
        self.guided_fp_x_tilde_right = self.encode2(self.local_x_tilde_right, reuse=True)

        self.new_x_tilde = self.incomplete_img + self.x_tilde * self.input_masks

        self.D_real_gan_logits = self.discriminator(self.input, self.local_input_left, self.local_input_right,
                                                    self.guided_fp_x_left, self.guided_fp_x_right, reuse=False)
        self.D_fake_gan_logits = self.discriminator(self.new_x_tilde, self.local_x_tilde_left, self.local_x_tilde_right,
                                                    self.guided_fp_x_tilde_left, self.guided_fp_x_tilde_right, reuse=True)

        if self.loss_type == 0:
            self.d_gan_loss = self.loss_hinge_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_hinge_gen(self.D_fake_gan_logits)
        elif self.loss_type == 1:
            self.d_gan_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_gen(self.D_fake_gan_logits)
        elif self.loss_type == 2:
            print "using lsgan"
            self.d_gan_loss = self.d_lsgan_loss(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.g_lsgan_loss(self.D_fake_gan_logits)

        #preception similar
        # self.fp_loss = 0.5 * self.cosine(self.guided_fp_x_left, self.guided_fp_x_tilde_left) + \
        #                 0.5 * self.cosine(self.guided_fp_x_right, self.guided_fp_x_tilde_right)
        self.fp_loss = tf.reduce_mean(tf.square(self.guided_fp_x_left - self.guided_fp_x_tilde_left)) + \
                            tf.reduce_mean(tf.square(self.guided_fp_x_right - self.guided_fp_x_tilde_right))

        #recon loss
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_tilde - self.input), axis=[1, 2, 3]) / (35 * 70 * self.channel))

        self.D_loss = self.d_gan_loss
        self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss + self.lam_fp * self.fp_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()

        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.ed_vars = [var for var in self.t_vars if 'ed' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]

        print "d_vars", len(self.d_vars)
        print "ed_vars", len(self.ed_vars)
        print "e_vars", len(self.e_vars)

        self.saver = tf.train.Saver()
        self.load_saver = tf.train.Saver(self.e_vars)
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def cosine(self, f1, f2):
        f1_norm = tf.nn.l2_normalize(f1, dim=0)
        f2_norm = tf.nn.l2_normalize(f2, dim=0)

        return tf.losses.cosine_distance(f1_norm, f2_norm, dim=0)

    #softplus
    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    #hinge loss
    #Hinge Loss + sigmoid
    def loss_hinge_dis(self, d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(self, d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    #lsgan
    def d_lsgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(self, d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    #wgan loss
    def d_wgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits)

    def g_wgan_loss(self, g_fake_logits):
        return - tf.reduce_mean(g_fake_logits)

    def gradient_penalty(self, x_tilde, x, local_x_tilde, local_x):
        self.differences1 = x_tilde - x
        self.differences2 = local_x_tilde - local_x
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates1 = x + self.alpha * self.differences1
        interpolates2 = local_x + self.alpha * self.differences2
        discri_logits = self.discriminator(interpolates1, interpolates2, reuse=True)
        gradients = tf.gradients(discri_logits, [interpolates1])[0]
        slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))

        gradients = tf.gradients(discri_logits, [interpolates2])[0]
        slopes2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))

        return tf.reduce_mean((slopes1 - 1.)**2 + (slopes2 - 1.)**2)

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver.restore(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(100000)))
            batch1, mask1, _, _, testbatch, testmask = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 200
            for j in range(batch_num):

                real_test_batch, real_test_mask = sess.run([testbatch, testmask])
                f_d = {self.input: real_test_batch, self.mask: real_test_mask}
                input_masks, test_incomplete_img, test_x_tilde, new_test_x_tilde = \
                    sess.run([self.input_masks, self.incomplete_img, self.x_tilde, self.new_x_tilde], feed_dict=f_d)
                test_output_concat = np.concatenate([real_test_batch, real_test_mask, test_incomplete_img, test_x_tilde,
                                                     new_test_x_tilde], axis=0)
                save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                        '{}/{:02d}_test_output.jpg'.format(self.test_sample_path, j))

            coord.request_stop()
            coord.join(threads)

    def test2(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver.restore(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(100000)))
            _, batch1, mask1, testbatch, testmask = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 3256 / self.batch_size
            for j in range(batch_num):

                real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)
                f_d = {self.input: real_test_batch, self.input_masks: batch_masks,
                       self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos}
                test_incomplete_img, test_x_tilde, test_new_x_tilde, local_input_left, local_input_right, input_masks, r_masks\
                    = sess.run([self.incomplete_img, self.x_tilde, self.new_x_tilde,
                                self.local_input_left, self.local_input_right, self.input_masks, 1 - self.input_masks], feed_dict=f_d)

                for i in range(self.batch_size):

                    save_images(np.reshape(input_masks[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_masks.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(r_masks[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_r_masks.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(real_test_batch[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_real.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(real_test_batch[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_real.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_incomplete_img[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_in_compelete.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_x_tilde[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_output.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_new_x_tilde[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_new_output.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(local_input_left[i], newshape=(1, self.output_size/2, self.output_size/2, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_local_input_left.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(local_input_right[i], newshape=(1, self.output_size/2, self.output_size/2, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_local_input_right.jpg'.format(self.test_sample_path, j, i))

            coord.request_stop()
            coord.join(threads)

    def test3(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver.restore(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(100000)))
            _, batch1, mask1, testbatch, testmask = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 1000 / self.batch_size
            for j in range(batch_num):

                real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)
                f_d = {self.input: real_test_batch, self.input_masks: batch_masks,
                       self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos}
                test_incomplete_img, test_x_tilde, test_new_x_tilde, local_input_left, local_input_right, input_masks, r_masks \
                    = sess.run([self.incomplete_img, self.x_tilde, self.new_x_tilde,
                                self.local_input_left, self.local_input_right, self.input_masks,
                                1 - self.input_masks], feed_dict=f_d)

                for i in range(self.batch_size):
                    save_images(
                        np.reshape(input_masks[i], newshape=(1, self.output_size, self.output_size, self.channel)),
                        [1, 1],
                        '{}/{:02d}_{:2d}_masks.jpg'.format(self.test_sample_path, j, i))
                    save_images(
                        np.reshape(r_masks[i], newshape=(1, self.output_size, self.output_size, self.channel)),
                        [1, 1],
                        '{}/{:02d}_{:2d}_r_masks.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(real_test_batch[i],
                                           newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_real.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(real_test_batch[i],
                                           newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_real.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_incomplete_img[i],
                                           newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_in_compelete.jpg'.format(self.test_sample_path, j, i))
                    save_images(
                        np.reshape(test_x_tilde[i], newshape=(1, self.output_size, self.output_size, self.channel)),
                        [1, 1],
                        '{}/{:02d}_{:2d}_output.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_new_x_tilde[i],
                                           newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_new_output.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(local_input_left[i],
                                           newshape=(1, self.output_size / 2, self.output_size / 2, self.channel)),
                                [1, 1],
                                '{}/{:02d}_{:2d}_local_input_left.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(local_input_right[i],
                                           newshape=(1, self.output_size / 2, self.output_size / 2, self.channel)),
                                [1, 1],
                                '{}/{:02d}_{:2d}_local_input_right.jpg'.format(self.test_sample_path, j, i))

            coord.request_stop()
            coord.join(threads)

    #@profile
    def train(self):

        opti_D = tf.train.AdamOptimizer(self.d_learning_rate * self.lr_decay,
                                         beta1=self.beta1, beta2=self.beta2).minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.g_learning_rate * self.lr_decay,
                                        beta1=self.beta1, beta2=self.beta2).minimize(loss=self.G_loss, var_list=self.ed_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0
            lr_decay = 1

            try:
                self.load_saver.restore(sess, os.path.join(self.read_model_path, 'model_{:06d}.ckpt'.format(100000)))
            except Exception as e:
                print("Model path may not be correct")

            print("Start read dataset")
            batch_image_path, batch_image, eye_pos, testbatch_image, test_eye_pos = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            real_test_batch, real_test_pos = sess.run([testbatch_image, test_eye_pos])

            while step <= self.max_iters:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                real_batch_image_path, real_batch_image, real_eye_pos = sess.run([batch_image_path, batch_image, eye_pos])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)

                f_d = {self.input: real_batch_image, self.input_masks: batch_masks,
                       self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos, self.lr_decay: lr_decay}

                # optimize D
                sess.run(opti_D, feed_dict=f_d)
                # optimize G
                sess.run(opti_G, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:

                    output_loss = sess.run([self.D_loss, self.G_loss, self.lam_recon * self.recon_loss, self.lam_fp * self.fp_loss], feed_dict=f_d)
                    print("step %d D_loss1=%.8f, G_loss=%.4f, Recon_loss=%.4f, Fp_loss=%.4f, lr_decay=%.4f" % (
                                         step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], lr_decay))

                if np.mod(step, 2000) == 0:
                    train_output_img = sess.run([self.local_input_left, self.local_input_right, self.incomplete_img, self.x_tilde,
                                                self.new_x_tilde, self.local_x_tilde_left, self.local_x_tilde_right], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)
                    #for test
                    f_d = {self.input: real_test_batch, self.input_masks: batch_masks,
                           self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos, self.lr_decay: lr_decay}
                    test_output_img = sess.run([self.incomplete_img, self.x_tilde, self.new_x_tilde], feed_dict=f_d)

                    output_concat = np.concatenate([real_batch_image,
                                                    train_output_img[2], train_output_img[3], train_output_img[4]], axis=0)
                    local_output_concat = np.concatenate([train_output_img[0],
                                                          train_output_img[1], train_output_img[5], train_output_img[6]], axis=0)
                    test_output_concat = np.concatenate([real_test_batch,
                                                         test_output_img[0], test_output_img[2], test_output_img[1]], axis=0)

                    save_images(local_output_concat, [local_output_concat.shape[0] / self.batch_size, self.batch_size],
                                '{}/{:02d}_local_output.jpg'.format(self.sample_path, step))
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                    save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_test_output.jpg'.format(self.sample_path, step))

                if np.mod(step, 20000) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print "Model saved in file: %s" % save_path

    def discriminator(self, incom_x, local_x_left, local_x_right, guided_fp_left, guided_fp_right, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse == True:
                scope.reuse_variables()

            x = incom_x
            for i in range(6):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                print output_dim
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_1_{}'.format(i)))
            x = tf.reshape(x, shape=[self.batch_size, -1])
            ful_global = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_fu1')

            x = tf.concat([local_x_left, local_x_right], axis=3)
            for i in range(5):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_2_{}'.format(i)))
            x = tf.reshape(x, shape=[self.batch_size, -1])
            ful_local = fully_connect(x, output_size=output_dim*2, use_sp=self.use_sp, scope='dis_fu2')

            ful = tf.concat([ful_global, ful_local, guided_fp_left, guided_fp_right], axis=1)
            ful = tf.nn.relu(fully_connect(ful, output_size=512, use_sp=self.use_sp, scope='dis_fu4'))
            gan_logits = fully_connect(ful, output_size=1, use_sp=self.use_sp, scope='dis_fu5')

            return gan_logits

    def encode_decode(self, input_x, img_mask, guided_fp_left, guided_fp_right, use_sp=False, reuse=False):

        with tf.variable_scope("ed") as scope:

            if reuse == True:
                scope.reuse_variables()
            #encode
            x = tf.concat([input_x, img_mask], axis=3)
            for i in range(6):
                c_dim = np.minimum(16 * np.power(2, i), 256)
                if i == 0:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=7, k_h=7, d_w=1, d_h=1, use_sp=use_sp, name='e_c{}'.format(i))
                                      , scope='e_in_{}'.format(i)))
                else:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=use_sp, name='e_c{}'.format(i))
                                      , scope='e_in_{}'.format(i)))

            bottleneck = tf.reshape(x, shape=[self.batch_size, -1])
            bottleneck = fully_connect(bottleneck, output_size=256, use_sp=use_sp, scope='e_ful1')
            bottleneck = tf.concat([bottleneck, guided_fp_left, guided_fp_right], axis=1)

            de_x = tf.nn.relu(fully_connect(bottleneck, output_size=256*8*8, use_sp=use_sp, scope='d_ful1'))
            de_x = tf.reshape(de_x, shape=[self.batch_size, 8, 8, 256])
            #de_x = tf.tile(de_x, (1, 8, 8, 1), name='tile')

            #decode
            for i in range(5):
                c_dim = np.maximum(256 / np.power(2, i), 16)
                output_dim = 16 * np.power(2, i)
                print de_x
                de_x = tf.nn.relu(instance_norm(de_conv(de_x, output_shape=[self.batch_size, output_dim, output_dim, c_dim], use_sp=use_sp,
                                                            name='g_deconv_{}'.format(i)), scope='g_in_{}'.format(i)))
            #de_x = tf.concat([de_x, input_x], axis=3)
            x_tilde1 = conv2d(de_x, output_dim=3, k_w=7, k_h=7, d_h=1, d_w=1, use_sp=use_sp, name='g_conv1')

            return tf.nn.tanh(x_tilde1)

    # new encoder
    def encode2(self, x, reuse=False):

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
            #rotation = fully_connect(bottleneck, output_size=1, scope='e_ful2')

            return content#, rotation

    def get_Mask_and_pos(self, eye_pos, flag=0):

        eye_pos = eye_pos
        #print eye_pos
        batch_mask = []
        batch_left_eye_pos = []
        batch_right_eye_pos = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            left_eye_pos = []
            right_eye_pos = []
            #eye
            if flag == 0:

                #left eye, y
                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_eye_pos[1] - 5 #current_eye_pos[3] / 2
                down_scale = current_eye_pos[1] + 30 #current_eye_pos[3] / 2
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_eye_pos[0] - 35 #current_eye_pos[2] / 2
                down_scale = current_eye_pos[0] + 35 #current_eye_pos[2] / 2
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
                left_eye_pos.append(float(l1_1)/self.output_size)
                left_eye_pos.append(float(l1_2)/self.output_size)
                left_eye_pos.append(float(u1_1)/self.output_size)
                left_eye_pos.append(float(u1_2)/self.output_size)

                #right eye, y
                scale = current_eye_pos[3] - 5 #current_eye_pos[7] / 2
                down_scale = current_eye_pos[3] + 30 #current_eye_pos[7] / 2

                l2_1 = int(scale)
                u2_1 = int(down_scale)

                #x
                scale = current_eye_pos[2] - 35 #current_eye_pos[6] / 2
                down_scale = current_eye_pos[2] + 35 #current_eye_pos[6] / 2
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







