import tensorflow as tf
import scipy.misc
import numpy as np
import glob
import os
import errno

class Dataset(object):

    def __init__(self, config):
        super(Dataset, self).__init__()
        self.data_dir = config.data_dir
        self.height, self.width, self.channel = config.hwc
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.capacity = config.capacity
        self.shuffle = config.shuffle

        print "Dataset.py Test code"
        print self.data_dir, self.height, self.width, self.channel
        print self.batch_size, self.num_threads, self.capacity, self.shuffle

        self.filenames1_list, self.mask1, self.filenames2_list, self.mask2 = self.readfilenames()
        self.train1_list, self.train_mask1_list = self.filenames1_list, self.mask1
        self.train2_list, self.train_mask2_list = self.filenames2_list[:-200], self.mask2[:-200]
        self.test_list, self.test_mask_list = self.filenames2_list[-200:], self.mask2[-200:]

        print len(self.filenames1_list), len(self.mask1), len(self.filenames2_list), len(self.mask2)
        print "Train1_list", len(self.train1_list), len(self.train_mask1_list)
        print "Train2_list", len(self.train2_list), len(self.train_mask2_list)
        print "Test_list", len(self.test_list), len(self.test_mask_list)

    def readfilenames(self):

        original_eye_path = 'eye_11_26_dataset'
        eye_mask = 'celeb_id_aligned_eye_mask_11_27'
        filenames1 = glob.glob(os.path.join(self.data_dir, original_eye_path + "/1/*.jpg"))

        print "original data 1", len(filenames1)

        mask1 = []
        middle_1 = []
        for i, pathname in enumerate(filenames1):
            newpathname = pathname.replace(original_eye_path, eye_mask)
            if os.path.exists(newpathname):
                mask1.append(newpathname)
            else:
                middle_1.append(pathname)
        for i, pathname in enumerate(middle_1):
            filenames1.remove(pathname)

        filenames2 = glob.glob(os.path.join(self.data_dir, original_eye_path + "/0/*.jpg"))

        print "original data 2", len(filenames2)

        mask2 = []
        middle_2 = []
        for j, pathname in enumerate(filenames2):
            newpathname = pathname.replace(original_eye_path, eye_mask)
            if os.path.exists(newpathname):
                mask2.append(newpathname)
            else:
                middle_2.append(pathname)
        for j, pathname in enumerate(middle_2):
            filenames2.remove(pathname)

        return filenames1, mask1, filenames2, mask2

    def read_images(self, input_queue, is_mask=False):

        content = tf.read_file(input_queue)
        image = tf.image.decode_jpeg(content, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, size=(self.height, self.width))

        if is_mask:
            return image / 255.0

        else:
            return image / 127.5 - 1.0

    def Get_queues_image_mask(self, list1, list2, is_shuffle, image_size=128):

        #convert list to tensor
        images1 = tf.convert_to_tensor(list1, dtype=tf.string)
        masks1 = tf.convert_to_tensor(list2, dtype=tf.string)

        #produce the list of tensor
        input_queue1 = tf.train.slice_input_producer([images1, masks1], shuffle=is_shuffle)

        images_queue1 = self.read_images(input_queue=input_queue1[0], is_mask=False)
        masks_queue1 = self.read_images(input_queue=input_queue1[1], is_mask=True)

        #method=1 nn methods
        images_queue1 = tf.image.resize_images(images_queue1, size=(image_size, image_size), method=1)
        masks_queue1 = tf.image.resize_images(masks_queue1, size=(image_size, image_size), method=1)

        return images_queue1, masks_queue1

    def input(self, image_size):

        images_queue1, masks_queue1 = self.Get_queues_image_mask(self.train1_list, self.train_mask1_list, is_shuffle=True, image_size=image_size)
        images_queue2, masks_queue2 = self.Get_queues_image_mask(self.train2_list, self.train_mask2_list, is_shuffle=True, image_size=image_size)
        images_queue3, masks_queue3 = self.Get_queues_image_mask(self.test_list, self.test_mask_list, is_shuffle=False, image_size=image_size)

        batch_image1, batch_mask1 = tf.train.shuffle_batch([images_queue1, masks_queue1],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=5000
                                                )
        batch_image2, batch_mask2 = tf.train.shuffle_batch([images_queue2, masks_queue2],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=5000
                                                )

        batch_test_image, batch_test_mask = tf.train.batch([images_queue3, masks_queue3],
                                                                   batch_size=self.batch_size,
                                                                   capacity=100,
                                                                   num_threads=1)

        return batch_image1, batch_mask1, batch_image2, batch_mask2, batch_test_image, batch_test_mask


def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(image, is_ouput=False):
    if is_ouput == True:
        print image[0]
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print result
    return result

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)