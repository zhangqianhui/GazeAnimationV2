from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_gan_losses_fn():
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def get_hinge_loss():
    def loss_hinge_dis(d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    return loss_hinge_dis, loss_hinge_gen

def get_softplus_loss():

    def loss_dis(d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    def loss_gen(d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    return loss_dis, loss_gen

def get_lsgan_loss():

    def d_lsgan_loss(d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) \
               + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    return d_lsgan_loss, g_lsgan_loss

def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def get_adversarial_loss(mode):

    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge':
        return get_hinge_loss()
    elif mode == 'lsgan':
        return get_lsgan_loss()
    elif mode == 'softplus':
        return get_softplus_loss()
    elif mode == 'wgan_gp':
        return get_wgan_losses_fn()
