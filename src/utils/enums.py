from enum import Enum


class Mode(Enum):
    TRAIN = "TRAIN"
    INFER = "INFER"

    def __str__(self):
        return self.value


class EmbeddingType(Enum):
    RANDOM = "RANDOM"
    WORD2VEC = "WORD2VEC"
    GLOVE = "GLOVE"

    def __str__(self):
        return self.value


class FieldIdentifier(Enum):
    ID = "id"
    TURN_1 = "turn_1"
    TURN_2 = "turn_2"
    TURN_3 = "turn_3"
    LABEL = "label"

    def __str__(self):
        return self.value
