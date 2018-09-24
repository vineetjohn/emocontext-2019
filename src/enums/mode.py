from enum import Enum


class Mode(Enum):
    TRAIN = "TRAIN"
    INFER = "INFER"

    def __str__(self):
        return self.value
