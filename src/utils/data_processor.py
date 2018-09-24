import logging
from torchtext import data
from src.config import global_config
from src.config.model_config import mconf

LOG = logging.getLogger(global_config.logger_name)


def read_data(file_path):
    LOG.debug("Reading data")

    id_field = data.Field()
    turn_1_field = data.Field(sequential=True)
    turn_2_field = data.Field(sequential=True)
    turn_3_field = data.Field(sequential=True)
    label_field = data.Field()

    train_data_fields = [
        ("id", id_field),
        ("turn1", turn_1_field),
        ("turn2", turn_2_field),
        ("turn3", turn_3_field),
        ("label", label_field),
    ]

    train_dataset = data.TabularDataset(
        path=file_path, format="TSV", fields=train_data_fields, skip_header=True
    )
    id_field.build_vocab(train_dataset)
    turn_1_field.build_vocab(train_dataset)
    turn_2_field.build_vocab(train_dataset)
    turn_3_field.build_vocab(train_dataset)
    label_field.build_vocab(train_dataset)

    iterator = data.Iterator(
        dataset=train_dataset, batch_size=mconf.batch_size, train=True, repeat=False
    )

    return iterator
