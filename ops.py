import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

#leakyRelu
def lrelu(x, alpha= 0.2):
    return tf.maximum(x , alpha*x)

#conv2d
def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h= 2, d_w=2, stddev=0.02,
           name="conv2d", use_sp=False, padding='SAME'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if use_sp != True:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

#avgpool
def downscale2d(x, k=2):

    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                    padding='VALID')

#dilated conv2d
def dilated_conv2d(input_, output_dim,
           k_h=3, k_w=3, stddev=0.02, rate=2,
           name="conv2d", use_sp=False, padding='SAME'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if use_sp != True:
            conv = tf.nn.atrous_conv2d(input_, w, rate=rate, padding=padding)
        else:
            conv = tf.nn.atrous_conv2d(input_, spectral_norm(w), rate=rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset

def weight_normalization(weight, scope='weight_norm'):

  """based upon openai's https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py"""
  weight_shape_list = weight.get_shape().as_list()
  if len(weight.get_shape()) == 2: #I think you want to sum on axis [0,1,2]
    g_shape = [weight_shape_list[1]]
  else:
    raise ValueError('dimensions unacceptable for weight normalization')

  with tf.variable_scope(scope):
    g = tf.get_variable('g_scalar', shape=g_shape, initializer = tf.ones_initializer())
    weight = g * tf.nn.l2_normalize(weight, dim=0)

    return weight

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

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, name=None, stddev=0.02, use_sp=True,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(name or "Linear"):
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

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)

def batch_normal(input, scope="scope", reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse, fused=True, updates_collections=None)

def Residual(x, output_dims=256, kernel=3, residual_name='resi'):

    with tf.variable_scope('residual_{}'.format(residual_name)):

        conv1 = instance_norm(dilated_conv2d(x, output_dims, k_h=kernel, k_w=kernel, name="conv1"), scope='in1')
        conv2 = instance_norm(dilated_conv2d(tf.nn.relu(conv1), output_dims, k_h=kernel, k_w=kernel, name="conv2"), scope='in2')
        resi = x + conv2

        return tf.nn.relu(resi)


NO_OPS = 'NO_OPS'
def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

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
            trainable=False
        )

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

def squeeze2d(x, factor=2):

    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])
    return x

def unsqueeze2d(x, factor=2):

    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    return x