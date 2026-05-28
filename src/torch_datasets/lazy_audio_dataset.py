from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf

class LazyAudioDataset(Dataset):
    '''
    LazyLoader dataset class, takes in a data directory and a target sampling rate.
    Used to lazyload audio files since the dataset is large
    '''
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        target_sample_rate: int = 16000,
    ):
        self.df = df.reset_index(drop=True)
        self.target_sample_rate = target_sample_rate
        self.audio_dir = Path(audio_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["file name"]

        start_sec = row.get("start_sec", 0.0)

        stem = Path(filepath).stem
        segments = sorted(list(self.audio_dir.glob(f"{stem}__segment*.wav")))
        if not segments:
            raise FileNotFoundError(f"Missing audio fragments for session stem: '{stem}' in {self.audio_dir}")

        nr_segments = len(segments) if segments else 1
        total_duration = 60 * nr_segments
        end_sec = row.get("end_sec", total_duration)

        segment_waveforms = []
        for file in segments:
            waveform_np, sample_rate = sf.read(file, dtype="float32")
            waveform = torch.from_numpy(waveform_np)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            segment_waveforms.append(waveform)
        
        combined_waveforms = torch.cat(segment_waveforms, dim=-1)
        start = int(start_sec * self.target_sample_rate)
        end = int(end_sec * self.target_sample_rate)
        combined_waveforms = combined_waveforms[..., start:end]

        return combined_waveforms
