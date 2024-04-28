import json
import random
from os import cpu_count

import numpy as np
import pandas as pd
from datasets import Dataset


def change_speed(df: pd.DataFrame, factor: float = None) -> tuple[pd.DataFrame, float]:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + np.random.random() * change_range

    df.start /= factor
    df.end /= factor
    df.duration = df.end - df.start
    return df, factor


def pitch_shift(df: pd.DataFrame, shift_threshold: int = 5) -> tuple[pd.DataFrame, int]:
    # No more than given number of steps
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


def apply_pitch_shift(batch: dict, augmentation_probability: float, augmentation_repetitions: int):
    """
    Takes a batch of size 1 and applies augmentation - to use with dataset.map
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


def apply_speed_change(batch: dict, augmentation_probability: float, augmentation_repetitions: int):
    """
    Takes a batch of size 1 and applies augmentation - to use with dataset.map
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


def augment_dataset(dataset: Dataset, augmentation_probability: float, augmentation_repetitions: int = 0):
    """
    Augment the dataset with dataset.map method using all cpus.

    If augmentation_cfg.repetitions is 0, will output a copy of the dataset.
    """
    if augmentation_repetitions == 0:
        return dataset

    num_cpus = cpu_count()

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
