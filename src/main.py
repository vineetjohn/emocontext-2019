import sys
import logging
import argparse
import torch

from src.config import global_config as gconf
from src.config.model_config import mconf
from src.utils import log_helper, dataset_helper, embedding_helper
from src.utils.enums import Mode, EmbeddingType, FieldIdentifier
from src.models.model import EmotionClassifier

LOG = logging.getLogger()


class Options(argparse.Namespace):
    def __init__(self):
        self.log_level = "INFO"
        self.train_file_path = None
        self.mode = None


def main(args):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str)
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), required=True)
    parser.parse_known_args(args=args, namespace=options)

    LOG = log_helper.setup_custom_logger(gconf.logger_name, options.log_level)

    LOG.debug("Options: {}".format(options))
    LOG.info("Running model in {} mode".format(options.mode))

    data_processor = dataset_helper.DataProcessor(options.train_file_path)
    train_iterator = data_processor.get_data_iterator()

    word_embedding = embedding_helper.WordEmbedding(
        vocab_size=len(data_processor.turn_3_field.vocab),
        dimensions=gconf.word_embedding_size,
    )
    classifier = EmotionClassifier(
        len(data_processor.label_field.vocab), word_embedding.get_embedder()
    )

    train_iterator.init_epoch()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=mconf.learning_rate)
    optimizer.zero_grad()
    for i, train_data in enumerate(train_iterator):
        LOG.debug("Processing iteration {}".format(i))
        print(train_data.TURN_3)
        logits = classifier(train_data.TURN_3[0], train_data.TURN_3[1])
        # print(logits)


if __name__ == "__main__":
    main(sys.argv[1:])
