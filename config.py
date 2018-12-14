import os

class Config:

    @property
    def base_path(self):
        return os.path.abspath(os.curdir)

    @property
    def data_dir(self):
        data_dir = os.path.join(self.base_path, '/home/?/dataset/')
        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join(self.base_path, 'train_log' + self.operation_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def read_model_path(self):
        model_path = os.path.join(self.exp_dir, 'read_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def write_model_path(self):
        model_path = os.path.join(self.exp_dir, 'write_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    training_step = 1
    is_training = False
    gpu_id = 0
    image_size = 256
    hwc = [image_size, image_size, 3]
    is_load = 0
    load_model = 0

    #loss_type
    max_iters = 100000
    g_learning_rate = 0.0001
    d_learning_rate = 0.0005
    loss_type = 0

    # hyper
    batch_size = 2
    lam_recon = 10
    lam_fp = 0
    weight_decay = 5e-5
    use_sp = True
    beta1 = 0.5
    beta2 = 0.999

    # dataset
    num_threads = 10
    capacity = 3000
    shuffle = True

    if training_step == 0:

        @property
        def sample_path(self):
            sample_path = os.path.join(self.exp_dir, 'sample_img')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            return sample_path

        @property
        def test_sample_path(self):
            sample_path = os.path.join(self.exp_dir, 'test_sample_img')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            return sample_path

        operation_name = "11_2_22"

    #for inpainting
    else:

        @property
        def sample_path(self):
            sample_path = os.path.join(self.exp_dir, 'sample_img_inpainting6')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            return sample_path

        @property
        def write_model_path(self):
            model_path = os.path.join(self.exp_dir, 'write_model6')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            return model_path

        @property
        def test_sample_path(self):
            sample_path = os.path.join(self.exp_dir, 'test_img_inpainting6_star')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            return sample_path

        operation_name = "12_13_1"
