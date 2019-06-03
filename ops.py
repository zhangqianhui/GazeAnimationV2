import tensorflow as tf


def lrelu(x, alpha= 0.2):
    return tf.maximum(x , alpha*x)


def conv2d(input_, output_dim,k_h=5,
           k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", use_sp=False, padding='SAME', output_b=False):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if use_sp != True:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        if output_b:
            return conv, biases
        else:
            return conv

def dilated_conv2d(input_, output_dim,
           k_h=3, k_w=3, stddev=0.02, rate=2,
           name="conv2d", use_sp=True, padding='SAME'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if use_sp != True:
            conv = tf.nn.atrous_conv2d(input_, w, rate=2, padding=padding)
        else:
            conv = tf.nn.atrous_conv2d(input_, spectral_norm(w), rate=rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def instance_norm(input, scope="instance_norm"):
    with tf.variable_scope(scope):
        depth = input.get_shape()[-1]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset


def de_conv(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, use_sp=False,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))

        if use_sp:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w), output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def fully_connect(input_, output_size, scope=None, stddev=0.02, use_sp=True,
                  bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start))

        if use_sp:
            mul = tf.matmul(input_, spectral_norm(matrix))
        else:
            mul = tf.matmul(input_, matrix)

        if with_w:
            return mul + bias, matrix, bias
        else:
            return mul + bias


def spectral_norm(W, collections=None, return_norm=False, name='sn'):
    shape = W.get_shape().as_list()

    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:

        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        else:
            _W = W

        u = tf.get_variable(
            name=name + "_u",
            shape=(_W.shape.as_list()[-1], shape[0]),
            initializer=tf.random_normal_initializer,
            collections=collections,
            trainable=False)

        _u = u

        for _ in range(1):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)

        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)

        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if return_norm:
        return W / sigma, sigma
    else:
        return W / sigma
