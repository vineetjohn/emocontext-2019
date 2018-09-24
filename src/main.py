import sys
import logging
import argparse
from src.config import global_config
from src.utils import log_helper, data_processor
from src.enums.mode import Mode

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

    LOG = log_helper.setup_custom_logger(global_config.logger_name, options.log_level)

    LOG.debug("Options: {}".format(options))
    LOG.info("Running model in {} mode".format(options.mode))

    train_iterator = data_processor.read_data(options.train_file_path)

    for i, train_data in enumerate(train_iterator):
        LOG.debug("Processing iteration {}".format(i))


if __name__ == "__main__":
    main(sys.argv[1:])
