import random
import numpy as np
import pandas as pd
import soundfile as sf
from typing import Any

from pipeline import ModelPipelineStep
from logger import console, logger

class AudioSplicer(ModelPipelineStep):
    name = "Audio Splicer"

    # x_test and y_test are just passed through so it doesnt break the pipeline
    inputs = {"x_train", "x_test", "y_train", "y_test"}
    outputs = {"x_train", "x_test", "y_train", "y_test"}

    def __init__(self, audio_path_col: str, chunk_duration: int = 15) -> None:
        super().__init__()

        self._chunk_duration = chunk_duration
        self._audio_col = audio_path_col

    def _get_random_slice(self, file_path: str) -> tuple[int, int]:
        try:
            with sf.SoundFile(
                    file_path, 
                    mode='r', 
                    samplerate=22050, 
                    channels=1, 
                    format='RAW', 
                    subtype='PCM_16'
                ) as f:
                    total_duration = int(f.frames / f.samplerate)
            if total_duration < self._chunk_duration:
                return 0, total_duration
            
            max_start = total_duration - self._chunk_duration
            start_point = round(random.uniform(0, max_start))
            return start_point, start_point + self._chunk_duration
        except Exception as e:
            logger.error(f"Error at getting splice for file at {file_path}: {e}")
            return 0, self._chunk_duration
    
    def _slice_over_df(self, df: pd.DataFrame, real_paths: dict = {}) -> pd.DataFrame:
        for id, row in df.iterrows():
            if pd.isna(row[self._audio_col]):
                options = real_paths.get(row["target"], [])
                if options:
                    picked_path = random.choice(options)
                    df.at[id, self._audio_col] = picked_path
            
            start, end = self._get_random_slice(row[self._audio_col])
            df.at[id, "start_sec"] = start
            df.at[id, "end_sec"] = end
        return df
                


    def run(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame,
            y_test: pd.Series,
            verbose: bool = True
            ) -> None:

        if verbose:
            console.section("Handling Class Imbalance on Audio Data")

        df_x = x_train.copy()
        df_x["target"] = y_train
        df_x["start_sec"] = None
        df_x["end_sec"] = None

        # check if smote ran previously
        smote_check = df_x[self._audio_col].isna().any()

        # save the real paths so synthetic rows can access them
        real_rows = df_x[df_x[self._audio_col].notna()].copy()
        real_paths = real_rows.groupby("target")[self._audio_col].apply(list).to_dict()

        if smote_check:
            if verbose:
                logger.info("SMOTE detected in previous step.\n"
                            "Creating splices over synthetic rows..."
                            )
        else:
            if verbose:
                logger.info("SMOTE not detected. \n"
                            f"Current class distribution: \n: {y_train.value_counts()}"
                            "\nCreating new synthetic rows..."
                            )
            class_counts = df_x["target"].value_counts()
            maj_class = class_counts.idxmax()
            min_class = class_counts.idxmin()
            difference = class_counts[maj_class] - class_counts[min_class]
            if difference > 0:
                minority_pool = df_x[df_x["target"] == min_class]
                synthetic_rows = minority_pool.sample(difference, replace=True).copy()
                df_x = pd.concat([df_x, synthetic_rows], ignore_index=True)
        
        df_x = self._slice_over_df(df_x, real_paths)
        df_x = df_x.sample(frac=1).reset_index(drop=True)

        x_resampled = df_x.drop(columns=["target"])
        y_resampled = df_x["target"]        

        if verbose:
            logger.info(f"Finised resampling of audio data. \n"
                        f"New class distribution: \n: {y_resampled.value_counts()}"
                        )
        return {
            "x_train": x_resampled,
            "x_test": x_test,
            "y_train": y_resampled,
            "y_test": y_test,
            }