import sys

import argparse
import dill
import logging
import os
import torch

from src.config import global_config as gconf
from src.config.model_config import mconf
from src.models.model import EmotionClassifier
from src.utils import log_helper, dataset_helper, embedding_helper
from src.utils.enums import Mode

LOG = logging.getLogger()


class Options(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.log_level = "INFO"
        self.train_file_path = None
        self.test_file_path = None
        self.mode = None
        self.epochs = 1
        self.model_directory = None


def train_model(options: Options):
    # Create directory to save training artifacts
    if not os.path.exists(gconf.MODEL_SAVE_DIR):
        os.makedirs(gconf.MODEL_SAVE_DIR)
    model_save_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_MODEL_FILENAME)
    embedding_save_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_EMBEDDING_FILENAME)

    # Define data pipeline and iterator
    data_processor = dataset_helper.TrainDataProcessor(options.train_file_path)
    train_iterator = data_processor.get_data_iterator()

    # Define word embedder
    word_embedding = embedding_helper.WordEmbedding(
        vocab_size=len(data_processor.turn_3_field.vocab),
        dimensions=gconf.WORD_EMBEDDING_SIZE,
    )

    # Define model
    model = EmotionClassifier(len(data_processor.label_field.vocab), word_embedding.get_embedder())

    # Initialization before training
    train_iterator.init_epoch()
    optimizer = torch.optim.Adam(model.parameters(), lr=mconf.learning_rate)
    optimizer.zero_grad()

    # Train model
    for epoch in range(options.epochs):
        for i, train_data in enumerate(train_iterator):
            # run encoder over 3rd utterance
            class_probs = model(train_data.TURN_3[0], train_data.TURN_3[1])

            # flattens from [1, batch_size] to [batch_size]
            labels = train_data.LABEL

            # compute NLL Loss
            loss = torch.nn.functional.cross_entropy(class_probs, labels)
            LOG.info("Loss: {:.2f}, Epoch: {}-{}".format(loss, epoch + 1, i + 1))

            loss.backward()
            optimizer.step()

        LOG.info("Saving model to disk ...")
        torch.save(obj=model.state_dict(), f=model_save_path, pickle_module=dill)
        torch.save(obj=model.embedding, f=embedding_save_path, pickle_module=dill)
        LOG.info("Saved model to %s", model_save_path)


def infer_emotion(options: Options):
    gconf.MODEL_SAVE_DIR = options.model_directory

    saved_vocab_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_VOCAB_FILENAME)
    vocab_field = torch.load(f=saved_vocab_path, pickle_module=dill)
    LOG.info("Loaded vocab")

    saved_label_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_LABEL_FILENAME)
    label_field = torch.load(f=saved_label_path, pickle_module=dill)
    LOG.info("Loaded labels")

    saved_embedding_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_EMBEDDING_FILENAME)
    embedding = torch.load(f=saved_embedding_path, pickle_module=dill)
    LOG.info("Loaded embeddings")

    saved_model_path = os.path.join(gconf.MODEL_SAVE_DIR, gconf.SAVED_MODEL_FILENAME)
    model = EmotionClassifier(len(label_field.vocab), embedding)
    model.load_state_dict(torch.load(f=saved_model_path, pickle_module=dill))
    model.eval()
    LOG.info("Loaded model")

    # Define data pipeline and iterator
    data_processor = dataset_helper.TestDataProcessor(options.test_file_path, vocab_field)
    test_iterator = data_processor.get_data_iterator()

    index_predictions = torch.LongTensor([])
    for i, test_data in enumerate(test_iterator):
        LOG.info("Predicting samples from index: {}".format(i * mconf.batch_size))
        class_probs = model(test_data.TURN_3[0], test_data.TURN_3[1])
        _, predictions = torch.max(class_probs, 1)

        index_predictions = torch.cat([index_predictions, predictions])
    class_predictions = [label_field.vocab.itos[x] for x in index_predictions]

    if not os.path.exists(gconf.OUTPUT_DIR):
        os.makedirs(gconf.OUTPUT_DIR)
    predictions_file_path = os.path.join(gconf.OUTPUT_DIR, gconf.PREDICTIONS_FILENAME)
    with open(predictions_file_path, 'w') as predictions_file:
        for prediction in class_predictions:
            predictions_file.write("{}\n".format(prediction))
    LOG.info("Predictions written to file {}".format(predictions_file_path))


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
