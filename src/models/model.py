import torch
from src.config import global_config as gconf
from src.config.model_config import mconf


class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_labels, embedding, layer_count=1, dropout=0):
        super(EmotionClassifier, self).__init__()
        self.layer_count = layer_count
        self.num_labels = num_labels
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = torch.nn.GRU(
            gconf.WORD_EMBEDDING_SIZE,
            mconf.gru_hidden_dim,
            layer_count,
            dropout=(0 if layer_count == 1 else dropout),
            bidirectional=True,
        )
        self.lin_1 = torch.nn.Linear(mconf.gru_hidden_dim * 2, num_labels, bias=True)
        self.softmax = torch.nn.LogSoftmax(1)
        self.loss = torch.nn.NLLLoss()

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through GRU
        _, hidden = self.gru(packed, hidden)

        # Concatenate forward and backward hidden unit of the GRU
        encoded_sentence = torch.cat(tensors=(hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Project hidden layer into a (num-labels)-sized vector
        unscaled_logits = self.lin_1(encoded_sentence)

        # Compute softmax over the labels
        probabilities = self.softmax(unscaled_logits)

        # Return output and final hidden state
        return probabilities
