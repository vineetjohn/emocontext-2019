from enum import Enum


class StringEnum(Enum):
    def __str__(self):
        return self.value


class Mode(StringEnum):
    TRAIN = "TRAIN"
    INFER = "INFER"


class EmbeddingType(StringEnum):
    RANDOM = "RANDOM"
    WORD2VEC = "WORD2VEC"
    GLOVE = "GLOVE"


class FieldIdentifier(StringEnum):
    ID = "id"
    TURN_1 = "turn_1"
    TURN_2 = "turn_2"
    TURN_3 = "turn_3"
    LABEL = "label"
