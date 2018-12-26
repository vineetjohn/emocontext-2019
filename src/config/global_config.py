from datetime import datetime as dt

EXPERIMENT_TIMESTAMP = dt.now().strftime("%Y%m%d%H%M%S")
LOGGER_NAME = "emocontext"
WORD_EMBEDDING_SIZE = 100
MODEL_SAVE_DIR = "./models/{}/".format(EXPERIMENT_TIMESTAMP)
MODEL_FILENAME = "classifier.mdl"
