import torch
import torch.nn as nn


class FusionLSTMModel(
    nn.Module,
):
    def __init__(self, lstm_layers: tuple, cur_input_size: int, ff_hidden_size, layers):
        super().__init__()

        seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout = lstm_layers

        # LSTM Component
        self.lstm = nn.LSTM(seq_input_size, lstm_hidden_size, lstm_num_layers, dropout=lstm_dropout, batch_first=True)

        # Tabular component
        self.embeddings_model = nn.Sequential(*layers)

        # Feed-forward Component
        self.ff = nn.Sequential(nn.Linear(lstm_hidden_size + cur_input_size, ff_hidden_size), nn.ReLU(), nn.Linear(ff_hidden_size, 1))

    def forward(self, historical_data: torch.Tensor, current_data: torch.Tensor):
        lstm_output, _ = self.lstm(historical_data)

        tabular_embedding_output = self.embeddings_model(current_data)

        # Select the last LSTM output (final time step)
        lstm_last_output = lstm_output[:, -1, :]

        # Concatenate the LSTM output with current features
        combined_features = torch.cat([lstm_last_output, tabular_embedding_output], dim=1)

        # Pass through the feed-forward component for final prediction
        result = self.ff(combined_features)
        return result


