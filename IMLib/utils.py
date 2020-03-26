from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imageio
import scipy.misc as misc
import scipy
import numpy as np
import cv2

def save_as_gif(images_list, out_path, gif_file_name='all', save_image=False):

    if os.path.exists(out_path) == False:
        os.mkdir(out_path)
    # save as .png
    if save_image == True:
        for n in range(len(images_list)):
            file_name = '{}.png'.format(n)
            save_path_and_name = os.path.join(out_path, file_name)
            misc.imsave(save_path_and_name, images_list[n])
    # save as .gif
    out_path_and_name = os.path.join(out_path, '{}.gif'.format(gif_file_name))
    imageio.mimsave(out_path_and_name, images_list, 'GIF', duration=0.1)

def get_image(image_path, crop_size=128, is_crop=False, resize_w=140, is_grayscale=False):
    return transform(imread(image_path , is_grayscale), crop_size, is_crop, resize_w)

def transform(image, crop_size=64, is_crop=True, resize_w=140):

    image = scipy.misc.imresize(image, [resize_w, resize_w])
    if is_crop:
        cropped_image = center_crop(image, crop_size)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h, crop_w=None):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)
    if rate < 0.5:
        x = np.fliplr(x)

    return x[j:j+crop_h, i:i+crop_w]

def transform_image(image):
    return (image + 1) * 127.5

def save_images(images, image_path, is_verse=True):
    if is_verse:
        return imageio.imwrite(image_path, inverse_transform(images))
    else:
        return imageio.imwrite(image_path, images)

def resizeImg(img, size=list):
    return scipy.misc.imresize(img, size)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    if len(images.shape) == 3:
        img = images
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

def inverse_transform(image):
    result = ((image + 1) * 127.5).astype(np.uint8)
    return result

height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0

def from_gaze2d(gaze, output_size, scale=1.0):

    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    r = int(height_to_eyeball_radius_ratio * oh_2)
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_radius = eyeball_radius_to_iris_diameter_ratio * eyeball_radius
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([
        -iris_distance * sin_phi * cos_theta,
        iris_distance * sin_theta,
    ])
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset
    angle = np.degrees(np.arctan2(iris_offset[1], iris_offset[0]))
    ellipse_max = eyeball_radius_to_iris_diameter_ratio * iris_radius
    ellipse_min = np.abs(ellipse_max * cos_phi * cos_theta)
    #gazemap = np.zeros((oh, ow), dtype=np.float32)

    # Draw eyeball
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv2.ellipse(gazemap, box=(iris_centre, (ellipse_min, ellipse_max), angle),
                         color = 1.0 , thickness=-1, lineType=cv2.LINE_AA)
    #outout = cv2.circle(test_gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv2.circle(gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    return np.asarray(gazemaps)


if __name__ == "__main__":

    target_angles = [[0.0, -1.0], [1.0, -1.0], [1.0, -0.66], [1.0, -0.33], [1.0, 0.0], [1.0, 0.33],
                     [1.0, 0.66], [1.0, 1.0], [0, 1.0], [-1.0, 1.0]]

    for i, angles in enumerate(target_angles):
        x, y = angles[0],  - 1 * angles[1]
        gazemaps = from_gaze2d((x, y), (64, 64))
        cv2.imwrite("gazemaps_{}.jpg".format(i), gazemaps[1,...] * 255.0)