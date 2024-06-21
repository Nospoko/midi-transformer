import json
import random
from os import cpu_count

import numpy as np
import pandas as pd
from datasets import Dataset


def change_speed(df: pd.DataFrame, factor: float = None) -> tuple[pd.DataFrame, float]:
    """
    Change the speed of the MIDI notes in the DataFrame by a given factor.
    If no factor is provided, a random factor within a specified range is used.

    Parameters:
        df (pd.DataFrame): DataFrame containing MIDI notes.
        factor (float, optional): Factor by which to change the speed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, float]: The modified DataFrame and the factor used.
    """
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + np.random.random() * change_range

    df.start /= factor
    df.end /= factor
    df.duration = df.end - df.start
    return df, factor


def pitch_shift(df: pd.DataFrame, shift_threshold: int = 5) -> tuple[pd.DataFrame, int]:
    """
    Shift the pitch of the MIDI notes in the DataFrame by a random amount within the given threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing MIDI notes.
        shift_threshold (int): Maximum number of semitones to shift.

    Returns:
        tuple[pd.DataFrame, int]: The modified DataFrame and the shift amount used.
    """
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, df.pitch.min() - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - df.pitch.max())

    if low_shift > high_shift:
        shift = 0
        print("Pitch shift edge case:", df.pitch.min(), df.pitch.max())
    else:
        shift = np.random.randint(low=low_shift, high=high_shift + 1)
    df.pitch += shift
    return df, shift


def apply_pitch_shift(batch: dict, augmentation_probability: float, augmentation_repetitions: int) -> dict:
    """
    Apply pitch shift augmentation to a batch of MIDI notes with a given probability.

    Parameters:
        batch (dict): Batch of MIDI notes.
        augmentation_probability (float): Probability of applying augmentation.
        augmentation_repetitions (int): Number of times to repeat the augmentation.

    Returns:
        dict: Augmented batch of MIDI notes.
    """
    assert len(batch["notes"]) == 1
    source = json.loads(batch["source"][0])
    notes = batch["notes"][0]
    df = pd.DataFrame(notes)
    for _ in range(augmentation_repetitions):
        if random.random() < augmentation_probability:
            augmented, shift = pitch_shift(df=df.copy())
            batch["notes"].append(augmented.to_dict(orient="series"))
            batch["source"].append(json.dumps(source | {"pitch_shift": shift}))

    return batch


def apply_speed_change(batch: dict, augmentation_probability: float, augmentation_repetitions: int) -> dict:
    """
    Apply speed change augmentation to a batch of MIDI notes with a given probability.

    Parameters:
        batch (dict): Batch of MIDI notes.
        augmentation_probability (float): Probability of applying augmentation.
        augmentation_repetitions (int): Number of times to repeat the augmentation.

    Returns:
        dict: Augmented batch of MIDI notes.
    """
    assert len(batch["notes"]) == 1
    source = json.loads(batch["source"][0])
    notes = batch["notes"][0]
    df = pd.DataFrame(notes)
    for _ in range(augmentation_repetitions):
        if random.random() < augmentation_probability:
            augmented, factor = change_speed(df=df.copy())
            batch["notes"].append(augmented.to_dict(orient="series"))
            batch["source"].append(json.dumps(source | {"change_speed_factor": factor}))

    return batch


def augment_dataset(dataset: Dataset, augmentation_probability: float, augmentation_repetitions: int = 0) -> Dataset:
    """
    Augment the dataset by applying pitch shift and speed change augmentations using all available CPUs.

    Parameters:
        dataset (Dataset): Dataset to augment.
        augmentation_probability (float): Probability of applying augmentation.
        augmentation_repetitions (int, optional): Number of times to repeat the augmentation. Defaults to 0.

    Returns:
        Dataset: Augmented dataset.
    """
    if augmentation_repetitions == 0:
        return dataset

    num_cpus = cpu_count() - 4  # Use all CPUs except 4

    augmentation_arguments = {
        "augmentation_probability": augmentation_probability,
        "augmentation_repetitions": augmentation_repetitions,
    }
    dataset = dataset.map(
        apply_pitch_shift,
        fn_kwargs=augmentation_arguments,
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    dataset = dataset.map(
        apply_speed_change,
        fn_kwargs=augmentation_arguments,
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    return dataset
