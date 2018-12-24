class ModelConfig:
    def __init__(self):
        # batch settings
        self.batch_size = 32
        self.learning_rate = 0.001

        # gru settings
        self.gru_hidden_dim = 50

    def init_from_dict(self, previous_config):
        for key in previous_config:
            setattr(self, key, previous_config[key])


mconf = ModelConfig()
