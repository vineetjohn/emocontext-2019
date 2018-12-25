import sys
import logging
import argparse
import torch

from src.config import global_config as gconf
from src.config.model_config import mconf
from src.utils import log_helper, dataset_helper, embedding_helper
from src.utils.enums import Mode
from src.models.model import EmotionClassifier

LOG = logging.getLogger()


class Options(argparse.Namespace):
    def __init__(self):
        self.log_level = "INFO"
        self.train_file_path = None
        self.mode = None
        self.epochs = 1


def main(args):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str)
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), required=True)
    parser.add_argument("--epochs", type=int)
    parser.parse_known_args(args=args, namespace=options)

    global LOG
    LOG = log_helper.setup_custom_logger(gconf.logger_name, options.log_level)

    LOG.debug("Options: %s", options)
    LOG.info("Running model in %s mode", options.mode)

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

    for epoch in range(options.epochs):
        for i, train_data in enumerate(train_iterator):
            # run encoder over 3rd utterance
            class_probs = classifier(train_data.TURN_3[0], train_data.TURN_3[1])

            # flattens from [1, batch_size] to [batch_size]
            labels = train_data.LABEL.reshape((train_data.LABEL.size()[1]))

            # compute NLL Loss
            loss = torch.nn.functional.cross_entropy(class_probs, labels)
            LOG.info("loss: {:.2f}, epoch: {}-{}".format(loss, epoch + 1, i + 1))

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main(sys.argv[1:])
