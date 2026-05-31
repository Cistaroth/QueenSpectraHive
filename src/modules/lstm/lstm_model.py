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
        super().__init__()

        seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout = lstm_layers

        # Normalize MFCC features per timestep before the LSTM
        self.input_norm = nn.LayerNorm(seq_input_size)

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        # Tabular component
        self.embeddings_model = nn.Sequential(
            *embeddings_model
        )

        # Feed-forward Component
        self.ff = nn.Sequential(
            nn.Linear(lstm_hidden_size + features_input_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, 1)
        )

    def forward(
        self,
        historical_data: torch.Tensor,
        current_data: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ):
        lstm_output, _ = self.lstm(self.input_norm(historical_data))

        tabular_embedding_output = self.embeddings_model(current_data)

        # Mean-pool the LSTM outputs over the REAL (unpadded) timesteps. This is robust
        # to the very long MFCC sequences (the last hidden state forgets early audio)
        # and, with the length mask, never averages in zero-padded steps — unlike the
        # previous ``lstm_output[:, -1, :]`` which could read padding for short clips.
        if lengths is not None:
            max_len = lstm_output.size(1)
            mask = (
                torch.arange(max_len, device=lstm_output.device)[None, :]
                < lengths.to(lstm_output.device)[:, None]
            ).unsqueeze(-1).to(lstm_output.dtype)
            summed = (lstm_output * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            seq_representation = summed / counts
        else:
            seq_representation = lstm_output.mean(dim=1)

        # Concatenate the pooled sequence representation with current features
        combined_features = torch.cat([seq_representation, tabular_embedding_output], dim=1)

        # Pass through the feed-forward component for final prediction
        result = self.ff(combined_features)
        return result


