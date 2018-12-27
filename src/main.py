import os
import sys
import logging
import argparse
import torch
from typing import Any

from src.config import global_config as gconf
from src.config.model_config import mconf
from src.utils import log_helper, dataset_helper, embedding_helper
from src.utils.enums import Mode
from src.models.model import EmotionClassifier

LOG = logging.getLogger()


class Options(argparse.Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.log_level = "INFO"
        self.train_file_path = None
        self.mode = None
        self.epochs = 1
        self.model_directory = None


def train_model(options):
    # Define data pipeline and iterator
    data_processor = dataset_helper.DataProcessor(options.train_file_path)
    train_iterator = data_processor.get_data_iterator()

    # Define word embedder
    word_embedding = embedding_helper.WordEmbedding(
        vocab_size=len(data_processor.turn_3_field.vocab),
        dimensions=gconf.WORD_EMBEDDING_SIZE,
    )

    # Define model
    model = EmotionClassifier(
        len(data_processor.label_field.vocab), word_embedding.get_embedder()
    )

    # Initialization before training
    train_iterator.init_epoch()
    optimizer = torch.optim.Adam(model.parameters(), lr=mconf.learning_rate)
    optimizer.zero_grad()

    # Create directory to save training artifacts
    if not os.path.exists(gconf.MODEL_SAVE_DIR):
        os.makedirs(gconf.MODEL_SAVE_DIR)
    model_save_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.MODEL_FILENAME)

    # Train model
    for epoch in range(options.epochs):
        for i, train_data in enumerate(train_iterator):
            # run encoder over 3rd utterance
            class_probs = model(train_data.TURN_3[0], train_data.TURN_3[1])

            # flattens from [1, batch_size] to [batch_size]
            labels = train_data.LABEL.reshape((train_data.LABEL.size()[1]))

            # compute NLL Loss
            loss = torch.nn.functional.cross_entropy(class_probs, labels)
            LOG.info("Loss: {:.2f}, Epoch: {}-{}".format(loss, epoch + 1, i + 1))

            loss.backward()
            optimizer.step()

        LOG.info("Saving model to disk ...")
        torch.save(model.state_dict(), model_save_path)
        LOG.info("Saved model to %s", model_save_path)


def infer_emotion(options):
    pass


def main(args):
    # Read command line options
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), required=True)
    parser.parse_known_args(args=args, namespace=options)

    if options.mode == Mode.TRAIN:
        parser.add_argument("--train-file-path", type=str, required=True)
        parser.add_argument("--epochs", type=int)
    elif options.mode == Mode.INFER:
        parser.add_argument("--model-directory", type=str, required=True)
        parser.add_argument("--test-file-path", type=str, required=True)
        parser.add_argument("--predictions-save-path", type=str, required=True)

    parser.parse_known_args(args=args, namespace=options)

    # Initialize global logger
    global LOG
    LOG = log_helper.setup_custom_logger(gconf.LOGGER_NAME, options.log_level)

    LOG.debug("Options: %s", options)
    LOG.info("Running model in %s mode", options.mode)

    if options.mode == Mode.TRAIN:
        train_model(options)
    elif options.mode == Mode.INFER:
        infer_emotion(options)

    LOG.info("Run complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
