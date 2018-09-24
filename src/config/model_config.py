class ModelConfig:
    def __init__(self):
        # batch settings
        self.batch_size = 32

    def init_from_dict(self, previous_config):
        for key in previous_config:
            setattr(self, key, previous_config[key])


mconf = ModelConfig()
