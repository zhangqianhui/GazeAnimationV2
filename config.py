import os


class Config:

    exp_name = "4_19_6"

    CUDA = True
    gpu_id = 0

    training_dataset = 0
    # model configuration
    image_size = 256
    hwc = [image_size, image_size, 3]
    pos_number = 4
    use_sp = True
    is_supervised = True

    # training configuration
    is_training = True
    shuffle = True
    batch_size = 8
    max_iters = 500000
    beta1 = 0.5
    beta2 = 0.999
    g_learning_rate = 0.0001
    d_learning_rate = 0.0001
    lam_percep = 1
    lam_recon = 1
    lam_ss = 1
    learning_rate_init = 1
    loss_type = 1

    capacity = 5000
    num_threads = 10

    pretrain_model_index = 100000
    #  input directories
    image_dirname = "image_dataname"
    pretrain_model_dirname = "sg_pre_model_g/"
    attr_0_filename = "eye_test.txt"
    attr_1_filename = "eye_train.txt"
    basepath = "/home/*/dataset"  # your NewGazeDataset Path

    if training_dataset:
        data_dirname = "clumbia"
    else:
        data_dirname = 'NewGazeData'

    # output directories names
    model_dirname = "models"
    result_dirname = "results"
    sample_dirname = "samples"
    test_dirname = "testresults"
    log_dirname = "logs"

    @property
    def base_path(self):
        #return os.path.abspath(os.curdir)
        return self.basepath

    @property
    def dataset_dir(self):
        dataset_dir = os.path.join(self.base_path, self.data_dirname)
        if not os.path.exists(dataset_dir):
            raise ValueError('Please specify a data dir.')
        return dataset_dir

    @property
    def cur_dir(self):
        exp_dir = os.path.abspath(os.curdir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join(self.cur_dir, 'train_log' + self.exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def pretrain_model_dir(self):
        model_path = os.path.join(self.cur_dir, self.pretrain_model_dirname)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def model_dir(self):
        model_path = os.path.join(self.exp_dir, self.model_dirname)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, self.log_dirname)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def evaluation_path(self):
        evaluation_path = os.path.join(self.exp_dir, 'validate_evaluation_result')
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        return evaluation_path

    @property
    def sample_dir(self):
        sample_path = os.path.join(self.exp_dir, self.sample_dirname)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    @property
    def result_dir(self):
        sample_path = os.path.join(self.exp_dir, self.result_dirname)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    @property
    def testresult_dir(self):
        testresult_dir = os.path.join(self.exp_dir, self.test_dirname)
        if not os.path.exists(testresult_dir):
            os.makedirs(testresult_dir)
        return testresult_dir

    @property
    def MODEL_DIR(self):
        #model_path = os.path.join(os.path.dirname(self.base_path),'A_SAEGAN_1/image_net')
        model_path = './imagenet/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path




