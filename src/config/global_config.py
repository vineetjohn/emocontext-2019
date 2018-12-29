from datetime import datetime as dt

EXPERIMENT_TIMESTAMP = dt.now().strftime("%Y%m%d%H%M%S")
LOGGER_NAME = "emocontext"
WORD_EMBEDDING_SIZE = 100
MODEL_SAVE_DIR = "./models/{}/".format(EXPERIMENT_TIMESTAMP)

SAVED_MODEL_FILENAME = "classifier"
SAVED_VOCAB_FILENAME = "vocab"
SAVED_LABEL_FILENAME = "label"

OUTPUT_DIR = "./output/{}/".format(EXPERIMENT_TIMESTAMP)
PREDICTIONS_FILENAME = "predictions.txt"
