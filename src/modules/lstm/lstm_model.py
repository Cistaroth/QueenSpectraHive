import torch
import torch.nn as nn


class FusionLSTMModel(
    nn.Module,
):
    def __init__(
        self,
        lstm_layers: tuple,
        features_input_size: int,
        embeddings_model: tuple,
        ff_hidden_size: int,
    ) -> None:
        """
        Initialize the FusionLSTMModel.

        Args:
            lstm_layers (tuple): A tuple containing the LSTM layer parameters in the order (
                input_size, hidden_size, num_layers, dropout).
            features_input_size (int): The size of the input features for the tabular data.
            embeddings_model (tuple): A tuple of nn.Module layers that process the tabular data.
            ff_hidden_size (int): The hidden layer size for the feed-forward network after concatenation
        Returns:
            None
        """
        super().__init__()

        seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout = lstm_layers

        # Normalization for sequential data
        self.input_norm = nn.LayerNorm(seq_input_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Tabular
        self.embeddings_model = nn.Sequential(*embeddings_model)

        # Feed-forward layers for combined features
        self.ff = nn.Sequential(
            nn.Linear(lstm_hidden_size + features_input_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, 1),
        )

    def forward(
        self,
        historical_data: torch.Tensor,
        current_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            historical_data (torch.Tensor): Tensor of shape (batch_size, seq_len, seq_input
                _size) containing the sequential data (e.g., MFCC features).
            current_data (torch.Tensor): Tensor of shape (batch_size, features_input_size)
                containing the tabular data features.
            lengths (torch.Tensor | None): Optional tensor of shape (batch_size,) containing
                the actual lengths of the sequences in historical_data for proper masking.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing the predicted probabilities.
        """
        lstm_output, _ = self.lstm(self.input_norm(historical_data))

        tabular_embedding_output = self.embeddings_model(current_data)

        seq_representation = lstm_output.mean(dim=1)

        combined_features = torch.cat(
            [seq_representation, tabular_embedding_output], dim=1
        )

        result = self.ff(combined_features)
        return result
