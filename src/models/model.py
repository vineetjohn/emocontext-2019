import torch
from torch import nn
from torch.nn import functional as F

from src.config import global_config as gconf
from src.config.model_config import mconf


class EmotionClassifierRNN(nn.Module):
    def __init__(self, num_labels: int, embedding: nn.Embedding):
        super(EmotionClassifierRNN, self).__init__()
        self.num_labels = num_labels
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(
            input_size=gconf.WORD_EMBEDDING_SIZE,
            hidden_size=mconf.rnn_hidden_dim,
            num_layers=mconf.rnn_layers,
            dropout=(0 if mconf.rnn_layers == 1 else mconf.rnn_dropout),
            bidirectional=True,
        )
        self.linear_1 = nn.Linear(in_features=mconf.rnn_hidden_dim * 2, out_features=num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, input_lengths):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through GRU
        _, hidden = self.gru(packed, None)

        # Concatenate forward and backward hidden unit of the GRU
        encoded_sentence = torch.cat(tensors=(hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Project hidden layer into a (num-labels)-sized vector
        unscaled_logits = self.linear_1(encoded_sentence)

        # Compute softmax over the labels
        probabilities = self.softmax(unscaled_logits)

        # Return output and final hidden state
        return probabilities


class EmotionClassifierCNN(nn.Module):
    def __init__(self, num_labels: int, embedding: nn.Embedding):
        super(EmotionClassifierCNN, self).__init__()
        self.num_labels = num_labels
        self.embedding = embedding

        # Initialize CNN
        self.conv_nets = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=mconf.cnn_hidden_dim,
                kernel_size=(x, gconf.WORD_EMBEDDING_SIZE)
            ) for x in mconf.cnn_kernel_sizes
        ])
        self.linear_1 = nn.Linear(
            in_features=mconf.cnn_hidden_dim * len(mconf.cnn_kernel_sizes),
            out_features=num_labels
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, input_lengths):
        while input_seq.size(0) < max(mconf.cnn_kernel_sizes):
            input_seq = torch.cat([torch.tensor(input_seq), torch.tensor(input_seq)])

        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq).permute(1, 0, 2).unsqueeze(1)

        # Convolve over the embedded input
        hidden_tensors = [F.relu(conv_net(embedded).squeeze(3)) for conv_net in self.conv_nets]

        # Max pooling
        pooled_tensors = [F.max_pool1d(hidden, hidden.size()[2]).squeeze(2) for hidden in hidden_tensors]

        # Apply dropout
        encoded = F.dropout(torch.cat(pooled_tensors, dim=1), mconf.lin_dropout)

        # Project hidden layer into a (num-labels)-sized vector
        unscaled_logits = self.linear_1(encoded)

        # Compute softmax over the labels
        probabilities = self.softmax(unscaled_logits)

        # Return output and final hidden state
        return probabilities
