import logging
from torchtext import data
from src.config import global_config as gconf
from src.config.model_config import mconf

LOG = logging.getLogger(gconf.LOGGER_NAME)


class DataProcessor:
    def __init__(self, train_file_path):
        self.train_file_path = train_file_path

        self.id_field = data.Field()
        self.turn_1_field = data.Field(sequential=True)
        self.turn_2_field = data.Field(sequential=True)
        self.turn_3_field = data.Field(sequential=True, include_lengths=True)
        self.label_field = data.Field()

    def get_data_iterator(self):
        train_data_fields = [
            ("ID", self.id_field),
            ("TURN_1", self.turn_1_field),
            ("TURN_2", self.turn_2_field),
            ("TURN_3", self.turn_3_field),
            ("LABEL", self.label_field),
        ]

        train_dataset = data.TabularDataset(
            path=self.train_file_path,
            format="TSV",
            fields=train_data_fields,
            skip_header=True
        )
        self.id_field.build_vocab(train_dataset)
        self.turn_1_field.build_vocab(train_dataset)
        self.turn_2_field.build_vocab(train_dataset)
        self.turn_3_field.build_vocab(train_dataset)
        self.label_field.build_vocab(train_dataset)

        iterator = data.Iterator(
            dataset=train_dataset,
            batch_size=mconf.batch_size,
            train=True,
            repeat=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.TURN_3)
        )

        return iterator
