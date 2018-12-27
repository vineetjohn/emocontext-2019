import dill
import torch
import logging
from torchtext import data
from torchtext.data import Field

from src.config import global_config as gconf
from src.config.model_config import mconf

LOG = logging.getLogger(gconf.LOGGER_NAME)


class TrainDataProcessor:
    def __init__(self, train_file_path: str):
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

        # Save vocab and labels
        vocab_save_path = gconf.MODEL_SAVE_DIR + gconf.SAVED_VOCAB_FILENAME
        torch.save(obj=self.turn_3_field, f=vocab_save_path, pickle_module=dill)
        LOG.info("Saved turn 3 vocab")

        label_save_path = gconf.MODEL_SAVE_DIR + gconf.SAVED_LABEL_FILENAME
        torch.save(obj=self.label_field, f=label_save_path, pickle_module=dill)
        LOG.info("Saved label vocab")

        iterator = data.Iterator(
            dataset=train_dataset,
            batch_size=mconf.batch_size,
            train=True,
            repeat=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.TURN_3)
        )

        return iterator


class TestDataProcessor:
    def __init__(self, test_file_path: str, vocab_field: Field):
        self.test_file_path = test_file_path

        self.id_field = data.Field()
        self.turn_1_field = data.Field(sequential=True)
        self.turn_2_field = data.Field(sequential=True)
        self.turn_3_field = vocab_field

    def get_data_iterator(self):
        test_data_fields = [
            ("ID", self.id_field),
            ("TURN_1", self.turn_1_field),
            ("TURN_2", self.turn_2_field),
            ("TURN_3", self.turn_3_field),
        ]

        test_dataset = data.TabularDataset(
            path=self.test_file_path,
            format="TSV",
            fields=test_data_fields,
            skip_header=True
        )
        self.id_field.build_vocab(test_dataset)
        self.turn_1_field.build_vocab(test_dataset)
        self.turn_2_field.build_vocab(test_dataset)

        iterator = data.Iterator(
            dataset=test_dataset,
            batch_size=mconf.batch_size,
            train=False,
            repeat=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.TURN_3)
        )

        return iterator
