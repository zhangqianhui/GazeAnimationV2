import os
import scipy.misc
import numpy as np
import tensorflow as tf

class Dataset(object):

    def __init__(self, config):
        super(Dataset, self).__init__()

        self.dataset_dir = config.dataset_dir
        self.image_dirname = config.image_dirname
        self.attr_0_filename = config.attr_0_filename
        self.attr_1_filename = config.attr_1_filename
        self.height, self.width, self.channel = config.hwc
        # intput queue
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.shuffle = config.shuffle

        if config.training_dataset == 0:
            self.train_images_list, self.train_eye_pos, self.test_images_list, self.test_eye_pos, self.test_num = self.readfilenames()
        else:
            self.train_images_list, self.train_eye_pos,self.test_images_list, self.test_eye_pos, self.test_num = self.readfilenames_clumbia()
        #self.train_images_list, self.train_eye_pos, self.test_images_list, self.test_eye_pos, self.test_num = self.readfilenames()
        print "Numbers of dataset for training and testing", len(self.train_images_list), len(self.train_eye_pos), \
            len(self.test_images_list), len(self.test_eye_pos)

    def readfilenames(self):

        train_eye_pos = []
        train_images_list = []
        fh = open(os.path.join(self.dataset_dir, self.attr_1_filename))

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.dataset_dir, "1/"+filenames[0]+".jpg")):
                train_images_list.append(os.path.join(self.dataset_dir, "1/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                train_eye_pos.append(eye_pos)

        fh.close()

        fh = open(os.path.join(self.dataset_dir,self.attr_0_filename))
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.dataset_dir, "0/"+filenames[0]+".jpg")):
                test_images_list.append(os.path.join(self.dataset_dir,"0/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                test_eye_pos.append(eye_pos)
                #print test_eye_pos

        fh.close()
        return train_images_list, train_eye_pos, test_images_list, test_eye_pos, len(test_images_list)

    def readfilenames_clumbia(self):

        #dir_path = "waittocut/"
        fh = open(self.dataset_dir + "/eye_position.txt")
        train_images_list = []
        train_eye_pos = []
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 29)[0]
            all_item = f.split(' ', 29)
            file_id = filenames.split('_', 5)[0]
            if file_id == '0041' or file_id == '0042' or file_id == '0043' or file_id == '0044' or \
                file_id == '0045' or file_id == '0046' or file_id == '0047' or file_id == '0048' or \
                file_id == '0049' or file_id == '0050' or file_id == '0051' or file_id == '0052' or \
                file_id == '0053' or file_id == '0054' or file_id == '0055' or file_id == '0056':

                if os.path.exists(os.path.join(self.dataset_dir, file_id + "/" + filenames)):
                    eye_pos.extend([int(float(value)) for value in all_item[24:28]])
                    if len(eye_pos) == 4:
                        test_images_list.append(os.path.join(self.dataset_dir, file_id + "/" + filenames))

                        test_eye_pos.append(eye_pos)
            else:

                if os.path.exists(os.path.join(self.dataset_dir, file_id + "/" + filenames)):

                    eye_pos.extend([int(float(value)) for value in all_item[24:28]])
                    if len(eye_pos) == 4:
                        train_eye_pos.append(eye_pos)
                        train_images_list.append(os.path.join(self.dataset_dir, file_id + "/" + filenames))

        fh.close()

        return train_images_list, train_eye_pos, test_images_list, test_eye_pos, len(test_images_list)

    def read_images(self, input_queue):

        content = tf.read_file(input_queue)
        image = tf.image.decode_jpeg(content, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, size=(self.height, self.width))

        return image / 127.5 - 1.0

    def input(self):

        train_images = tf.convert_to_tensor(self.train_images_list, dtype=tf.string)
        train_eye_pos = tf.convert_to_tensor(self.train_eye_pos, dtype=tf.int32)
        train_queue = tf.train.slice_input_producer([train_images, train_eye_pos], shuffle=True)
        train_eye_pos_queue = train_queue[1]
        train_images_queue = self.read_images(input_queue=train_queue[0])

        test_images = tf.convert_to_tensor(self.test_images_list, dtype=tf.string)
        test_eye_pos = tf.convert_to_tensor(self.test_eye_pos, dtype=tf.int32)
        test_queue = tf.train.slice_input_producer([test_images, test_eye_pos], shuffle=False)
        test_eye_pos_queue = test_queue[1]
        test_images_queue = self.read_images(input_queue=test_queue[0])

        batch_path, batch_image1, batch_eye_pos1 = tf.train.shuffle_batch([train_queue[0], train_images_queue, train_eye_pos_queue],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=1000
                                                )

        batch_image2, batch_eye_pos2 = tf.train.batch([test_images_queue, test_eye_pos_queue],
                                                batch_size=self.batch_size,
                                                capacity=500,
                                                num_threads=1
                                                )

        return batch_path, batch_image1, batch_eye_pos1, batch_image2, batch_eye_pos2

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
