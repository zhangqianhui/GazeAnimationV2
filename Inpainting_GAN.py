import tensorflow as tf
from ops import conv2d, lrelu, instance_norm, de_conv, fully_connect, Residual, dilated_conv2d, upscale
from Dataset import save_images
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
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])

        self.domain_label = tf.placeholder(tf.int32, [self.batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self.incomplete_img = self.input * (1 - self.mask)
        self.x_tilde = self.encode_decode2(self.incomplete_img, 1 - self.mask, reuse=False)
        self.D_real_gan_logits = self.discriminator(self.input, self.input * self.mask, reuse=False)
        self.D_fake_gan_logits = self.discriminator(self.x_tilde, self.x_tilde * self.mask, reuse=True)

        self.d_gan_loss = self.loss_hinge_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
        self.g_gan_loss = self.loss_hinge_gen(self.D_fake_gan_logits)

        #recon loss
        self.recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.x_tilde - self.input), axis=[1, 2, 3]) / (
            self.output_size * self.output_size * self.channel))

        self.D_loss = self.d_gan_loss
        self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()

        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.ed_vars = [var for var in self.t_vars if 'ed' in var.name]

        print "d_vars", len(self.d_vars)
        print "ed_vars", len(self.ed_vars)

        self.saver = tf.train.Saver()
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

            batch_num = 750
            for j in range(batch_num):

                real_test_batch, real_test_mask = sess.run([testbatch, testmask])
                f_d = {self.input: real_test_batch, self.mask: real_test_mask}
                test_incomplete_img, test_x_tilde = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)

                for i in range(self.batch_size):

                    save_images(np.reshape(real_test_batch[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_real.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(real_test_mask[i], newshape=(1, self.output_size,self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_mask.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_incomplete_img[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_in_compelete.jpg'.format(self.test_sample_path, j, i))
                    save_images(np.reshape(test_x_tilde[i], newshape=(1, self.output_size, self.output_size, self.channel)), [1, 1],
                                '{}/{:02d}_{:2d}_output.jpg'.format(self.test_sample_path, j, i))

            coord.request_stop()
            coord.join(threads)
    #@profile
    def train(self):

        opti_D = tf.train.AdamOptimizer(self.d_learning_rate * self.lr_decay,
                                         beta1=self.beta1, beta2=self.beta2).minimize(loss=self.D_loss, var_list=self.d_vars)
        g_trainer = tf.train.AdamOptimizer(self.g_learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        g_gradients = g_trainer.compute_gradients(self.G_loss, var_list=self.ed_vars)
        opti_G = g_trainer.apply_gradients(g_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0
            lr_decay = 1

            print("Start read dataset")

            batch1, mask1, _, _, testbatch, testmask = self.data_ob.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            # try:
            #     self.load_saver.restore(sess, os.path.join(self.read_model_path, 'model_{:06d}.ckpt'.format(55000)))
            # except Exception as e:
            #     print("Model path may be not correct")
            real_test_batch, real_test_mask = sess.run([testbatch, testmask])
            while step <= self.max_iters:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                real_batch_1, real_mask1 = sess.run([batch1, mask1])
                f_d = {self.input: real_batch_1, self.mask: real_mask1, self.lr_decay: lr_decay}

                sess.run(opti_D, feed_dict=f_d)

                # optimize M
                sess.run(opti_G, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:
                    output_loss = sess.run(
                        [self.D_loss, self.G_loss, self.recon_loss],
                        feed_dict=f_d)
                    print("step %d D_loss1=%.4f, G_loss=%.4f, Recon_loss=%.4f, lr_decay=%.4f" % (
                        step, output_loss[0], output_loss[1], output_loss[2], lr_decay))

                if np.mod(step, 200) == 0:

                    incomplete_img, x_tilde = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)
                    #for test
                    f_d = {self.input: real_test_batch, self.mask: real_test_mask, self.lr_decay: lr_decay}
                    test_incomplete_img, test_x_tilde = sess.run([self.incomplete_img, self.x_tilde], feed_dict=f_d)

                    output_concat = np.concatenate([real_batch_1, real_mask1, incomplete_img, x_tilde], axis=0)
                    test_output_concat = np.concatenate([real_test_batch, real_test_mask, test_incomplete_img, test_x_tilde], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                    save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_test_output.jpg'.format(self.sample_path, step))

                if np.mod(step, 5000) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print "Model saved in file: %s" % save_path

    def discriminator(self, incom_x, local_x, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse == True:
                scope.reuse_variables()

            x = incom_x
            for i in range(6):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_1_{}'.format(i)))

            x = tf.reshape(x, shape=[self.batch_size, x.shape[1] * x.shape[2] * x.shape[3]])
            ful_local = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_fully1')

            x = local_x
            for i in range(6):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_2_{}'.format(i)))

            x = tf.reshape(x, shape=[self.batch_size, x.shape[1] * x.shape[2] * x.shape[3]])

            print x.shape

            ful_global = fully_connect(x, output_size=output_dim*4, use_sp=self.use_sp, scope='dis_fully2')
            gan_logits = fully_connect(tf.concat([ful_local, ful_global], axis=1), output_size=1, use_sp=self.use_sp, scope='dis_fully3')
            #fu2 = fully_connect(fu)

            return gan_logits

    def encode_decode(self, x, img_mask, reuse=False):

        with tf.variable_scope("ed") as scope:

            if reuse == True:
                scope.reuse_variables()

            x = tf.concat([x, img_mask], axis=3)
            conv1 = tf.nn.relu(
                instance_norm(conv2d(x, output_dim=32, k_w=7, k_h=7, d_w=1, d_h=1, use_sp=self.use_sp, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=64, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c3'), scope='e_in3'))
            conv4 = tf.nn.relu(
                instance_norm(conv2d(conv3, output_dim=256, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c4'), scope='e_in4'))

            r1 = Residual(conv4, residual_name='re_1')
            r2 = Residual(r1, residual_name='re_2')
            r3 = Residual(r2, residual_name='re_3')
            r4 = Residual(r3, residual_name='re_4')
            r5 = Residual(r4, residual_name='re_5')
            r6 = Residual(r5, residual_name='re_6')

            g_deconv2 = tf.nn.relu(instance_norm(de_conv(r6, output_shape=[self.batch_size, 32, 32, 256], k_w=4, k_h=4
                                                         , use_sp=self.use_sp, name='g_deconv2'), scope='gen_in2'))
            g_deconv3 = tf.nn.relu(instance_norm(de_conv(g_deconv2, output_shape=[self.batch_size, 64, 64, 128], k_w=4, k_h=4
                                                         , use_sp=self.use_sp, name='g_deconv3'), scope='gen_in3'))
            g_deconv4 = tf.nn.relu(instance_norm(de_conv(g_deconv3, output_shape=[self.batch_size, 128, 128, 64], k_w=4, k_h=4
                                                         , use_sp=self.use_sp, name='g_deconv4'), scope='gen_in4'))

            g_deconv4 = tf.concat([g_deconv4, x], axis=3)
            x_tilde1 = conv2d(g_deconv4, output_dim=3, k_w=7, k_h=7, d_h=1, d_w=1, use_sp=self.use_sp, name='gen_conv1')

            return tf.nn.tanh(x_tilde1)

    def encode_decode2(self, x, img_mask, reuse=False):

        with tf.variable_scope("ed") as scope:
            if reuse == True:
                scope.reuse_variables()

            x = tf.concat([x, img_mask], axis=3)
            conv1 = tf.nn.relu(
                instance_norm(conv2d(x, output_dim=32, k_w=7, k_h=7, d_w=1, d_h=1, use_sp=self.use_sp, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(instance_norm(conv2d(conv1, output_dim=64, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c2'),
                                             scope='e_in2'))
            conv3 = tf.nn.relu(instance_norm(conv2d(conv2, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c3'),
                                             scope='e_in3'))
            conv4 = tf.nn.relu(instance_norm(conv2d(conv3, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=self.use_sp, name='e_c4'),
                                             scope='e_in4'))

            bottleneck = tf.nn.relu(
                instance_norm(dilated_conv2d(conv4, output_dim=128, k_w=4, k_h=4, rate=2, use_sp=self.use_sp, name='e_c5'),
                              scope='e_in5'))
            bottleneck = tf.nn.relu(
                instance_norm(dilated_conv2d(bottleneck, output_dim=128, k_w=4, k_h=4, rate=2, use_sp=self.use_sp, name='e_c6'),
                              scope='e_in6'))
            bottleneck = tf.nn.relu(
                instance_norm(dilated_conv2d(bottleneck, output_dim=128, k_w=4, k_h=4, rate=2, use_sp=self.use_sp, name='e_c7'),
                              scope='e_in7'))

            g_conv2 = tf.nn.relu(instance_norm(conv2d(upscale(bottleneck, 2), output_dim=128, k_w=4, k_h=4, d_w=1, d_h=1, use_sp=self.use_sp
                                                      , name='g_conv2'), scope='gen_in2'))
            g_conv3 = tf.nn.relu(instance_norm(conv2d(upscale(g_conv2, 2), output_dim=64, k_w=4, k_h=4, d_w=1, d_h=1, use_sp=self.use_sp
                                                      , name='g_conv3'), scope='gen_in3'))
            g_conv4 = tf.nn.relu(instance_norm(conv2d(upscale(g_conv3, 2), output_dim=64, k_w=4, k_h=4, d_w=1, d_h=1, use_sp=self.use_sp
                                                      , name='g_conv4'), scope='gen_in4'))

            #g_conv4 = tf.concat([g_conv4, x], axis=3)
            x_tilde1 = conv2d(g_conv4, output_dim=3, k_w=7, k_h=7, d_h=1, d_w=1, use_sp=self.use_sp, name='gen_conv5')

            return tf.nn.tanh(x_tilde1)







