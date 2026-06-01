import random
from pathlib import Path
from typing import cast

import pandas as pd

from logger import console, logger
from pipeline import ModelPipelineStep


def random_audio_slice(
    file_path: str,
    audio_dir: Path | None,
    chunk_duration: int = 15,
    rng=None,
) -> tuple[int, int]:
    """
    Draw a random "chunk_duration"-second window over a recording's segments.

    Args:
        file_path (str): Path (or name) of the recording; only its stem is used to
            locate the matching segment files.
        audio_dir (Path | None): Directory holding the ``<stem>__segment*.wav`` files.
        chunk_duration (int, optional): Window length in seconds. Defaults to 15.
        rng (optional): A source exposing ``uniform(a, b)`` (e.g. ``random`` or a numpy
            ``Generator``) so callers can make sampling reproducible. Defaults to the
            global ``random`` module.
    Returns:
        tuple[int, int]: Start and end second offsets of the chosen window.
    """
    sampler = rng if rng is not None else random
    try:
        stem = Path(file_path).stem
        segments = list(audio_dir.glob(f"{stem}__segment*.wav")) if audio_dir else []
        nr_segments = len(segments) if segments else 1
        total_duration = 60 * nr_segments

        if total_duration < chunk_duration:
            return 0, total_duration

        max_start = total_duration - chunk_duration
        start_point = round(sampler.uniform(0, max_start))
        return start_point, start_point + chunk_duration
    except Exception as e:
        logger.error(f"Error at getting splice for file at {file_path}: {e}")
        return 0, chunk_duration


class AudioSplicerModule(ModelPipelineStep):
    name = "Audio Splicer"

    inputs = {"x_train", "y_train"}
    outputs = {"x_train", "y_train"}

    def __init__(
        self,
        audio_path_col: str,
        audio_dir: Path,
        chunk_duration: int = 15,
        oversample: bool = True,
    ) -> None:
        """
        Initializes the AudioSplicer class.

        Args:
            audio_path_col (str): The column name identifying audio file tracks.
            audio_dir (Path): The path to the audio files
            chunk_duration (int, optional): The target slice window in seconds. Defaults to 15.
            oversample (bool, optional): If True, balance classes by duplicating minority-class
                rows with replacement (legacy behaviour). If False, leave the class distribution
                untouched and rely on a downstream weighting scheme (e.g. pos_weight in
                BCEWithLogitsLoss) to handle imbalance. Defaults to True.
        Returns:
            None
        """
        super().__init__()

        self._chunk_duration = chunk_duration
        self._audio_col = audio_path_col
        self._audio_dir = audio_dir
        self._oversample = oversample

    def _get_random_slice(self, file_path: str) -> tuple[int, int]:
        """
        Calculates a randomized slice window over an assumed 60-second audio duration.

        Args:
            file_path (str): The location path of the audio asset.
        Returns:
            tuple[int, int]: Start and end integer coordinates in seconds.
        """
        return random_audio_slice(file_path, self._audio_dir, self._chunk_duration)

    def _slice_over_df(
        self, df: pd.DataFrame, existing_stems: set, real_paths: dict | None = None
    ) -> pd.DataFrame:
        """
        Iterates over the dataset to apply audio paths and coordinate offsets.

        Args:
            df (pd.DataFrame): Combined feature-target data pool.
            real_paths (dict, optional): Mapping of class labels to available real file lists.
        Returns:
            pd.DataFrame: Dataframe containing audio tracking metadata.
        """
        if real_paths is None:
            real_paths = {}

        for raw_id, row in df.iterrows():
            id = cast(int, raw_id)
            if (
                pd.isna(df.loc[id, self._audio_col])
                or Path(df.loc[id, self._audio_col]).stem not in existing_stems
            ):
                options = real_paths.get(row["target"], [])
                if options:
                    picked_path = random.choice(options)
                    df.at[id, self._audio_col] = picked_path

            full_path = Path(str(df.at[id, self._audio_col]))
            if self._audio_dir and not full_path.is_absolute():
                full_path = Path(self._audio_dir) / full_path
            resolved_path_str = str(full_path)
            df.at[id, self._audio_col] = resolved_path_str

            start, end = self._get_random_slice(resolved_path_str)
            df.at[id, "start_sec"] = start
            df.at[id, "end_sec"] = end
        return df

    def run(
        self, x_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        Executes randomized slicing coordinates and balances class targets.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame | pd.Series]: Dictionary containing matched multimodal sets.
        """
        if verbose:
            console.section("Handling Class Imbalance on Audio Data")
            logger.info(f"Indexing existing audio files inside {self._audio_dir}")

        # check existing files
        existing_stems = set()
        for p in self._audio_dir.glob("*__segment*.wav"):
            file_stem = p.name.split("__segment")[0]
            existing_stems.add(file_stem)

        df_x = x_train.copy()
        df_x["target"] = y_train
        df_x["start_sec"] = None
        df_x["end_sec"] = None

        # check if smote ran previously
        smote_check = df_x[self._audio_col].isna().any()

        # save the real and existing paths so synthetic rows can access them
        real_rows = df_x[
            df_x[self._audio_col].notna()
            & df_x[self._audio_col].apply(lambda x: Path(str(x)).stem in existing_stems)
        ].copy()
        real_paths = real_rows.groupby("target")[self._audio_col].apply(list).to_dict()

        if smote_check:
            if verbose:
                logger.info(
                    "SMOTE detected in previous step.\n"
                    "Creating splices over synthetic rows..."
                )
        elif not self._oversample:
            if verbose:
                logger.info(
                    "Oversampling disabled. \n"
                    f"Keeping original class distribution: \n: {y_train.value_counts()}"
                    "\nClass imbalance should be handled downstream "
                    "(e.g. pos_weight in BCEWithLogitsLoss)."
                )
        else:
            if verbose:
                logger.info(
                    "SMOTE not detected. \n"
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

        df_x = self._slice_over_df(df_x, existing_stems, real_paths)
        df_x = df_x.sample(frac=1).reset_index(drop=True)

        x_resampled = df_x.drop(columns=["target"])
        y_resampled = df_x["target"]

        if verbose:
            logger.info(
                f"Finished resampling of audio data. \n"
                f"New class distribution: \n: {y_resampled.value_counts()}"
            )
        return {
            "x_train": x_resampled,
            "y_train": y_resampled,
        }
