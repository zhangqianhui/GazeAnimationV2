import tensorflow as tf
from ops import conv2d, lrelu, instance_norm, fully_connect, dilated_conv2d, upscale, downscale2d
from Dataset2 import save_images
import os
import numpy as np

class Inpainting_GAN(object):

    # build model
    def __init__(self, data_ob, config, model_write, model_read, sample_path, pg, is_trans, base_step):

        print "Model.py Test"
        self.batch_size = config.batch_size
        print "batch_size", self.batch_size
        self.max_iters = config.max_iters  #min(config.max_iters * (2 ** (pg - 1)), 20000)
        print "max_iters", self.max_iters
        self.all_iters = config.all_iters
        self.base_step = base_step
        self.data_ob = data_ob
        print "data_ob", self.data_ob
        self.test_sample_path = config.test_sample_path
        print "test_sample_path", self.test_sample_path
        self.log_dir = config.log_dir
        print "log_dir", self.log_dir
        self.g_learning_rate = config.g_learning_rate
        print "g_learning_rate", self.g_learning_rate
        self.d_learning_rate = config.d_learning_rate
        print "d_learning_rate", self.d_learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        print "channel", self.channel
        self.lam_recon = config.lam_recon
        print "lam_recon", self.lam_recon
        self.lam_fp = config.lam_fp
        print "lam_fp", self.lam_fp
        self.beta1 = config.beta1
        print "beta1", self.beta1
        self.beta2 = config.beta2
        print "beta2", self.beta2
        self.use_sp = config.use_sp
        print "use_sp", self.use_sp
        self.loss_type = config.loss_type
        self.n_critic = config.n_critic

        #pg
        self.model_write = model_write
        print "model_write", self.model_write
        self.model_read = model_read
        print "model_read", self.model_read
        self.sample_path = sample_path
        print "sample_path", self.sample_path
        self.pg = pg
        print "pg", self.pg
        self.is_trans = is_trans
        print "is_trans", self.is_trans
        self.output_size = 4 * pow(2, self.pg - 1)
        print "output_size", self.output_size
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        print "input", self.input.shape
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        print "mask", self.mask.shape

        self.domain_label = tf.placeholder(tf.int32, [self.batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')
        self.alpha_trans = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')

    def build_model(self):

        if self.is_trans:
            self.lower_input = tf.image.resize_images(self.input,
                            size=[self.output_size/2, self.output_size/2], method=1)
            self.lower_input = tf.image.resize_images(self.lower_input,
                            size=[self.output_size, self.output_size], method=1)
            self.input = self.alpha_trans * self.input + (1 - self.alpha_trans) * self.lower_input
            self.input = tf.clip_by_value(self.input, -1, 1)

        self.incomplete_img = self.input * (1 - self.mask)
        self.x_tilde = self.encode_decode2(x=self.incomplete_img, img_mask= 1 - self.mask,
                                           pg=self.pg, is_trans=self.is_trans, alpha_trans=self.alpha_trans, reuse=False)
        #gan loss for data
        self.D_real_gan_logits = self.discriminator(self.input, self.input * self.mask, alpha_trans=self.alpha_trans,
                                                    pg=self.pg, is_trans=self.is_trans, reuse=False)
        self.D_fake_gan_logits = self.discriminator(self.x_tilde, self.x_tilde * self.mask, alpha_trans=self.alpha_trans,
                                                    pg=self.pg, is_trans=self.is_trans, reuse=True)

        if self.loss_type == 0:
            self.d_gan_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_gen(self.D_fake_gan_logits)
            self.d_gan_loss_original = self.d_gan_loss

        else:

            print "wgan_gp loss"
            self.d_gan_loss = self.d_wgan_loss(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.g_wgan_loss(self.D_fake_gan_logits)
            self.d_gan_loss_original = self.d_gan_loss
            self.d_gan_loss += 10 * self.gradient_penalty(self.x_tilde, self.input,
                                                          self.x_tilde * self.mask, self.input * self.mask)
        #recon loss
        self.recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.x_tilde - self.input), axis=[1, 2, 3]) / (
            self.output_size * self.output_size * self.channel))

        self.D_loss = self.d_gan_loss
        self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'dis' in var.name]
        for variable in self.d_vars:
            shape = variable.get_shape().as_list()
            print (variable.name, shape)

        self.ed_vars = [var for var in self.t_vars if 'ed' in var.name]
        for variable in self.ed_vars:
            shape = variable.get_shape().as_list()
            print (variable.name, shape)

        #save the variables which remains unchanged
        self.d_vars_c = [var for var in self.d_vars if 'dis_c' in var.name]
        self.g_vars_c = [var for var in self.ed_vars if 'gen_c' in var.name]

        #remove new variables for the new model
        self.d_vars_read = [var for var in self.d_vars_c if '{}'.format(self.output_size) not in var.name]
        self.g_vars_read = [var for var in self.g_vars_c if '{}'.format(self.output_size) not in var.name]

        #rgb variables
        self.d_vars_rgb = [var for var in self.d_vars if 'dis_rgb' in var.name]
        self.g_vars_rgb = [var for var in self.ed_vars if 'gen_rgb' in var.name]

        #remove new variables for the new model
        self.d_vars_rgb_read = [var for var in self.d_vars_rgb if '{}'.format(self.output_size) not in var.name]
        self.g_vars_rgb_read = [var for var in self.g_vars_rgb if '{}'.format(self.output_size) not in var.name]

        print "d_vars", len(self.d_vars)
        print "g_vars", len(self.ed_vars)
        print "d_vars_c", len(self.d_vars_c)
        print "g_vars_c", len(self.g_vars_c)
        print "d_vars_read", len(self.d_vars_read)
        print "g_vars_read", len(self.g_vars_read)
        print "d_vars_rgb", len(self.d_vars_rgb)
        print "g_vars_rgb", len(self.g_vars_rgb)
        print "d_vars_rgb_read", len(self.d_vars_rgb_read)
        print "g_vars_rgb_read", len(self.g_vars_rgb_read)

        self.saver = tf.train.Saver(self.d_vars + self.ed_vars)
        self.read_saver = tf.train.Saver(self.d_vars_read + self.g_vars_read)
        if len(self.d_vars_rgb_read + self.d_vars_rgb_read):
            self.rgb_saver = tf.train.Saver(self.d_vars_rgb_read + self.g_vars_rgb_read)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def cosine(self, f1, f2):
        f1_norm = tf.nn.l2_normalize(f1, dim=0)
        f2_norm = tf.nn.l2_normalize(f2, dim=0)

        return tf.losses.cosine_distance(f1_norm, f2_norm, dim=0)

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    #content adv
    def loss_content_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits))
        l2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits))
        return l1 + l2

    def loss_content_g(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=0.5 * tf.ones_like(d_fake_logits), logits=d_fake_logits))

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

        discri_logits = self.discriminator(interpolates1, interpolates2, alpha_trans=self.alpha_trans,
                                                    pg=self.pg, is_trans=self.is_trans, reuse=True)

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
                test_incomplete_img, test_x_tilde = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)
                test_output_concat = np.concatenate([real_test_batch, real_test_mask, test_incomplete_img, test_x_tilde], axis=0)
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
            batch1, mask1, _, _, testbatch, testmask = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = 200
            for j in range(batch_num):

                real_test_batch, real_test_mask = sess.run([testbatch, testmask])
                f_d = {self.input: real_test_batch, self.mask: real_test_mask}
                test_x_tilde = sess.run(self.x_tilde, feed_dict=f_d)
                save_images(test_x_tilde, [test_x_tilde.shape[0] / self.batch_size, self.batch_size],
                            '{}/{:02d}_test_output.jpg'.format(self.test_sample_path, j))

            coord.request_stop()
            coord.join(threads)

    #@profile
    def train(self):

        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_trans_assign = self.alpha_trans.assign(step_pl / self.max_iters)
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
            step = (self.base_step - 4) * self.max_iters
            max_iters = step + self.max_iters

            print step
            print max_iters
            lr_decay = 1
            print("Start read dataset")
            batch1, mask1, batch2, mask2, testbatch, testmask = self.data_ob.input(image_size=self.output_size)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            print "model_read", self.model_read

            print self.pg
            if self.pg != 3 and self.pg != 8:
                if self.is_trans:
                    print "read variables"
                    self.read_saver.restore(sess, self.model_read)
                    self.rgb_saver.restore(sess, self.model_read)
                else:
                    print "read all variables"
                    self.saver.restore(sess, self.model_read)

            real_test_batch, real_test_mask = sess.run([testbatch, testmask])
            real_sample_alpha = 1
            this_step = 0
            while step <= max_iters:

                lr_decay = (self.all_iters - step) / float(self.all_iters)
                for i in range(self.n_critic):
                    real_batch_1, real_mask1 = sess.run([batch1, mask1])
                    f_d = {self.input: real_batch_1, self.mask: real_mask1, self.lr_decay: lr_decay}
                    # optimize D
                    sess.run(opti_D, feed_dict=f_d)

                # optimize M
                sess.run(opti_G, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, this_step)

                sess.run(alpha_trans_assign, feed_dict={step_pl: this_step})

                if step % 100 == 0:

                    D_loss, D_loss_original, G_loss, Recon_loss, Alpha_tra = sess.run(
                        [self.D_loss, self.d_gan_loss_original, self.G_loss, self.recon_loss, self.alpha_trans],
                        feed_dict=f_d)
                    print("PG=%d step %d D_loss=%.4f, D_original=%.4f, G_loss=%.4f, Recon_loss=%.4f, lr_decay=%.4f, Alpha_tra=%.4f,"
                          "Real_sample_alpha=%.4f" % (self.pg,
                        step, D_loss, D_loss_original, G_loss, Recon_loss, lr_decay, Alpha_tra, real_sample_alpha))

                if np.mod(step, 1000) == 0:

                    incomplete_img1, x_tilde1 = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)
                    x_tilde1 = np.clip(x_tilde1, -1, 1)
                    #for test
                    f_d = {self.input: real_test_batch, self.mask: real_test_mask}
                    test_incomplete_img, test_x_tilde = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)

                    test_x_tilde = np.clip(test_x_tilde, -1, 1)
                    output_concat = np.concatenate([real_batch_1, real_mask1, incomplete_img1, x_tilde1], axis=0)
                    test_output_concat = np.concatenate([real_test_batch, real_test_mask, test_incomplete_img, test_x_tilde], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output3.jpg'.format(self.sample_path, step))
                    save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_test_output3.jpg'.format(self.sample_path, step))

                if np.mod(step, 10000) == 0 and step != 0:
                    self.saver.save(sess, self.model_write)

                step += 1
                this_step +=1

            print "model_write", self.model_write
            save_path = self.saver.save(sess, self.model_write)
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()

    def discriminator(self, incom_x, local_x, pg=1, is_trans=False, alpha_trans=0.01, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse == True:
                scope.reuse_variables()

            #global discriminator
            x = incom_x
            if is_trans:
                x_trans = downscale2d(x)
                #from rgb
                x_trans = lrelu(conv2d(x_trans, output_dim=self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_sp=self.use_sp,
                                       name='dis_rgb_g_{}'.format(x_trans.shape[1])))
            x = lrelu(conv2d(x, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_sp=self.use_sp,
                             name='dis_rgb_g_{}'.format(x.shape[1])))
            for i in range(pg - 1):
                x = lrelu(conv2d(x, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_sp=self.use_sp,
                                 name='dis_conv_g_{}'.format(x.shape[1])))
                x = downscale2d(x)
                if i == 0 and is_trans:
                    x = alpha_trans * x + (1 - alpha_trans) * x_trans
            x = lrelu(conv2d(x, output_dim=self.get_nf(1), k_h=3, k_w=3, d_h=1, d_w=1, use_sp=self.use_sp,
                             name='dis_conv_g_1_{}'.format(x.shape[1])))
            x = tf.reshape(x, [self.batch_size, -1])
            x_g = fully_connect(x, output_size=256, use_sp=self.use_sp, name='dis_conv_g_fully')

            #local discriminator
            x = local_x
            if is_trans:
                x_trans = downscale2d(x)
                #from rgb
                x_trans = lrelu(conv2d(x_trans, output_dim=self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_sp=self.use_sp,
                                       name='dis_rgb_l_{}'.format(x_trans.shape[1])))
            x = lrelu(conv2d(x, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_sp=self.use_sp,
                             name='dis_rgb_l_{}'.format(x.shape[1])))

            for i in range(pg - 1):
                x = lrelu(conv2d(x, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_sp=self.use_sp,
                                 name='dis_conv_l_{}'.format(x.shape[1])))
                x= downscale2d(x)
                if i == 0 and is_trans:
                    x = alpha_trans * x + (1 - alpha_trans) * x_trans

            x = lrelu(conv2d(x, output_dim=self.get_nf(1), k_h=3, k_w=3, d_h=1, d_w=1, use_sp=self.use_sp,
                             name='dis_conv_l_1_{}'.format(x.shape[1])))
            x = tf.reshape(x, [self.batch_size, -1])
            x_l = fully_connect(x, output_size=256, use_sp=self.use_sp, name='dis_conv_l_fully')

            logits = fully_connect(tf.concat([x_g, x_l], axis=1), output_size=1, use_sp=self.use_sp, name='dis_conv_fully')

            return logits

    def encode_decode2(self, x, img_mask, pg=1, is_trans=False, alpha_trans=0.01, reuse=False):

        with tf.variable_scope("ed") as scope:

            if reuse == True:
                scope.reuse_variables()

            x = tf.concat([x, img_mask], axis=3)
            if is_trans:
                x_trans = downscale2d(x)
                #fromrgb
                x_trans = tf.nn.relu(instance_norm(conv2d(x_trans, output_dim=self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1,
                                       name='gen_rgb_e_{}'.format(x_trans.shape[1])), scope='gen_rgb_e_in_{}'.format(x_trans.shape[1])))
            #fromrgb
            x = tf.nn.relu(instance_norm(conv2d(x, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_h=1, d_w=1,
                             name='gen_rgb_e_{}'.format(x.shape[1])), scope='gen_rgb_e_in_{}'.format(x.shape[1])))
            for i in range(pg - 1):
                print "encode", x.shape
                x = tf.nn.relu(instance_norm(conv2d(x, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1,
                                 name='gen_conv_e_{}'.format(x.shape[1])), scope='gen_conv_e_in_{}'.format(x.shape[1])))
                x = downscale2d(x)
                if i == 0 and is_trans:
                    x = alpha_trans * x + (1 - alpha_trans) * x_trans
            up_x = tf.nn.relu(
                instance_norm(dilated_conv2d(x, output_dim=512, k_w=3, k_h=3, rate=4, name='gen_conv_dilated'),
                              scope='gen_conv_in'))
            up_x = tf.nn.relu(instance_norm(conv2d(up_x, output_dim=self.get_nf(1), d_w=1, d_h=1, name='gen_conv_d'),
                                       scope='gen_conv_d_in_{}'.format(x.shape[1])))
            for i in range(pg - 1):

                print "decode", up_x.shape
                if i == pg - 2 and is_trans:
                    #torgb
                    up_x_trans = conv2d(up_x, output_dim=self.channel, k_w=1, k_h=1, d_w=1, d_h=1,
                                        name='gen_rgb_d_{}'.format(up_x.shape[1]))
                    up_x_trans = upscale(up_x_trans, 2)

                up_x = upscale(up_x, 2)
                up_x = tf.nn.relu(instance_norm(conv2d(up_x, output_dim=self.get_nf(i + 1), d_w=1, d_h=1,
                                name='gen_conv_d_{}'.format(up_x.shape[1])), scope='gen_conv_d_in_{}'.format(up_x.shape[1])))
            #torgb
            up_x = conv2d(up_x, output_dim=self.channel, k_w=1, k_h=1, d_w=1, d_h=1,
                        name='gen_rgb_d_{}'.format(up_x.shape[1]))
            if pg == 1: up_x = up_x
            else:
                if is_trans: up_x = (1 - alpha_trans) * up_x_trans + alpha_trans * up_x
                else:
                    up_x = up_x
            return up_x

    def get_nf(self, stage):
        return min(1024 / (2 ** (stage * 1)), 512)







