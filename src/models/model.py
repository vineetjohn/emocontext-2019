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
            gconf.word_embedding_size,
            mconf.gru_hidden_dim,
            layer_count,
            dropout=(0 if layer_count == 1 else dropout),
            bidirectional=True,
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = (
            outputs[:, :, : mconf.gru_hidden_dim]
            + outputs[:, :, mconf.gru_hidden_dim :]
        )

        # Return output and final hidden state
        return outputs, hidden
