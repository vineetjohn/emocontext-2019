class ModelConfig:
    def __init__(self):
        # batch settings
        self.batch_size = 32
        self.learning_rate = 0.0001

        # cnn settings
        self.cnn_hidden_dim = 50
        self.cnn_kernel_sizes = [3, 4, 5]

        # rnn settings
        self.rnn_hidden_dim = 50
        self.rnn_dropout = 0.8
        self.rnn_layers = 1

        # linear settings
        self.lin_dropout = 0.8

    def init_from_dict(self, previous_config):
        for key in previous_config:
            setattr(self, key, previous_config[key])


mconf = ModelConfig()
