import torch

from src.utils.enums import EmbeddingType


class WordEmbedding:
    def __init__(
        self, file_path=None, embedding_type=None, vocab_size=None, dimensions=None
    ):
        self.file_path = file_path
        self.embedding_type = embedding_type

        if bool(self.file_path) ^ bool(self.embedding_type):
            raise Exception(
                "Embedding file path and type should not be mutually exclusive"
            )
        if not embedding_type:
            self.embedding_type = EmbeddingType.RANDOM

        self.vocab_size = vocab_size
        self.dimensions = dimensions

    def get_embedder(self):
        if self.embedding_type == EmbeddingType.RANDOM:
            return torch.nn.Embedding(self.vocab_size, self.dimensions)
        else:
            raise Exception(
                "This type of embedding parsing ({}) is not implemented".format(
                    self.embedding_type
                )
            )
