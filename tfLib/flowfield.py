import tensorflow as tf


def bilinear_sample(input, flow, name):
    # reference to spatial transform network
    # 1.details can be found in office release:
    #   https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py
    # 2.maybe another good implement can be found in:
    #   https://github.com/kevinzakka/spatial-transformer-network/blob/master/transformer.py
    #   but this one maybe contain some problems, go to --> https://github.com/kevinzakka/spatial-transformer-network/issues/10
    with tf.variable_scope(name):
        N, iH, iW, iC = input.get_shape().as_list()
        _, fH, fW, fC = flow.get_shape().as_list()

        assert iH == fH and iW == fW
        # re-order & reshape: N,H,W,C --> N,C,H*W , shape= ( 16,2,3500 )
        flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [-1, fC, fH * fW])
        # get mesh-grid, 2,H*W
        indices_grid = meshgrid(iH, iW)
        transformed_grid = tf.add(flow, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])  # x_s should be (16,1,3500)
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])  # y_s should be ( 16,1,3500)
        # look tf.slice with ctrl , to figure out its meanning
        x_s_flatten = tf.reshape(x_s, [-1])  # should be (16*3500)
        y_s_flatten = tf.reshape(y_s, [-1])  # should be (16*3500)
        transformed_image = interpolate(input, x_s_flatten, y_s_flatten, iH, iW, 'interpolate')
        # print(transformed_image.get_shape().as_list())
        transformed_image = tf.reshape(transformed_image, [N, iH, iW, iC])

        return transformed_image

def meshgrid(height, width, ones_flag=None):

    with tf.variable_scope('meshgrid'):
        y_linspace = tf.linspace(-1., 1., height)
        x_linspace = tf.linspace(-1., 1., width)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=[-1])   #[H*W]
        y_coordinates = tf.reshape(y_coordinates, shape=[-1])   #[H*W]
        if ones_flag is None:
            indices_grid = tf.stack([x_coordinates, y_coordinates], axis=0) #[2, H*W]
        else:
            indices_grid = tf.stack([x_coordinates, y_coordinates, tf.ones_like(x_coordinates)], axis=0)

        return indices_grid


def interpolate(input, x, y, out_height, out_width, name):
    # parameters: input is input image,which has shape of (batchsize,height,width,3)
    # x,y is flattened coordinates , which has shape of (16*3500) = 56000
    # out_heigth,out_width = height,width
    with tf.variable_scope(name):
        N, H, W, C = input.get_shape().as_list()  #64, 40, 72, 3

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        H_f = tf.cast(H, dtype=tf.float32)
        W_f = tf.cast(W, dtype=tf.float32)
        # note that x,y belongs to [-1,1] before
        x = (x + 1.0) * (W_f - 1) * 0.5 # x now is [0,2]*0.5*[width-1],is [0,1]*[width-1]
                                        # shape 16 * 3500
        y = (y + 1.0) * (H_f - 1) * 0.5
        # get x0 and x1 in bilinear interpolation
        x0 = tf.cast(tf.floor(x), tf.int32) # cast to int ,discrete
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # clip the coordinate value
        max_y = tf.cast(H - 1, dtype=tf.int32)
        max_x = tf.cast(W - 1, dtype=tf.int32)
        zero = tf.constant([0], shape=(1,), dtype=tf.int32)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        # note x0,x1,y0,y1 have same shape 16 * 3500
        # go to method , look tf.clip_by_value,
        # realizing restrict op
        flat_image_dimensions = H * W
        pixels_batch = tf.range(N) * flat_image_dimensions
        # note N is batchsize, pixels_batch has shape [16]
        # plus, it's value is [0,1,2,...15]* 3500
        flat_output_dimensions = out_height * out_width
        # a scalar
        base = repeat(pixels_batch, flat_output_dimensions)
        # return 16 * 3500, go to see concrete value.

        base_y0 = base + y0 * W
        # [0*3500,.....1*3500,....2*3500,....]+[]
        base_y1 = base + y1 * W
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        # gather every pixel value
        flat_image = tf.reshape(input, shape=(-1, C))
        flat_image = tf.cast(flat_image, dtype=tf.float32)

        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        #for mask the interpolate part which pixel don't move
        mask = area_a + area_b + area_c + area_d
        output = (1 - mask) * flat_image + mask * output

        return output

def repeat(x, n_repeats):
    # parameters x: list [16]
    #            n_repeats : scalar,3500
    with tf.variable_scope('_repeat'):
        rep = tf.reshape(tf.ones(shape=tf.stack([n_repeats, ]), dtype=tf.int32), (1, n_repeats))
        # just know rep has shape (1,3500), and it's value is 1
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        # after reshape , matmul is (16,1)X(1,3500)
        # in matrix multi, result has shape ( 16,3500)
        # plus, in each row i, has same value  i * 3500
        return tf.reshape(x, [-1])  # return 16* 3500
