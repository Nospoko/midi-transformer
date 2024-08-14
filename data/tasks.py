import numpy as np
import pandas as pd


def add_noise_to_notes(notes: pd.DataFrame, attribute: str, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Add random noise to the specified attribute of the notes DataFrame.
    """
    if attribute not in ["velocity", "pitch", "start", "end", "time"]:
        raise ValueError("Attribute must be one of 'velocity', 'pitch', 'start', 'end', or 'time'")

    noisy_notes = notes.copy()

    if attribute in ["velocity", "pitch"]:
        attr_range = noisy_notes[attribute].max() - noisy_notes[attribute].min()
        noise = np.random.normal(0, noise_level * attr_range, len(noisy_notes))
        noisy_notes[attribute] += noise.astype(int)

        # Ensure values stay within valid ranges
        if attribute == "velocity":
            noisy_notes[attribute] = noisy_notes[attribute].clip(0, 127)
        elif attribute == "pitch":
            noisy_notes[attribute] = noisy_notes[attribute].clip(21, 109)

    elif attribute in ["start", "end"]:
        max_time = noisy_notes["end"].max()
        noise = np.random.normal(0, noise_level * max_time, len(noisy_notes))
        noisy_notes[attribute] += noise

        if attribute == "start":
            # Ensure start times are not negative and end times are after start times
            noisy_notes["start"] = noisy_notes["start"].clip(0)
            noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]

        elif attribute == "end":
            noisy_notes["end"].clip(0)
            noisy_notes["start"] = noisy_notes["end"] - noisy_notes["duration"]
            noisy_notes["start"].clip(0)

    elif attribute == "time":
        max_time = noisy_notes["end"].max()
        duration_range = noisy_notes["duration"].max() - noisy_notes["duration"].min()
        duration_noise = np.random.normal(0, noise_level * duration_range, len(noisy_notes))
        start_noise = np.random.normal(0, noise_level * max_time, len(noisy_notes))

        noisy_notes["start"] += start_noise
        noisy_notes["start"].clip(0)
        noisy_notes["duration"] += duration_noise
        noisy_notes["duration"].clip(0)

        noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]

    return noisy_notes


def add_comprehensive_noise(notes: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Add random noise to time, pitch, and velocity of the notes DataFrame simultaneously.
    """
    noisy_notes = notes.copy()

    # Add noise to velocity
    velocity_range = noisy_notes["velocity"].max() - noisy_notes["velocity"].min()
    velocity_noise = np.random.normal(0, noise_level * velocity_range, len(noisy_notes))
    noisy_notes["velocity"] += velocity_noise.astype(int)
    noisy_notes["velocity"] = noisy_notes["velocity"].clip(0, 127)

    # Add noise to pitch
    pitch_range = noisy_notes["pitch"].max() - noisy_notes["pitch"].min()
    pitch_noise = np.random.normal(0, noise_level * pitch_range, len(noisy_notes))
    noisy_notes["pitch"] += pitch_noise.astype(int)
    noisy_notes["pitch"] = noisy_notes["pitch"].clip(21, 109)

    # Add noise to time (start and duration)
    max_time = noisy_notes["end"].max()
    duration_range = noisy_notes["duration"].max() - noisy_notes["duration"].min()

    start_noise = np.random.normal(0, noise_level * max_time, len(noisy_notes))
    duration_noise = np.random.normal(0, noise_level * duration_range, len(noisy_notes))

    noisy_notes["start"] += start_noise
    noisy_notes["start"] = noisy_notes["start"].clip(0)
    noisy_notes["duration"] += duration_noise
    noisy_notes["duration"] = noisy_notes["duration"].clip(0)
    noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]

    return noisy_notes


def comprehensive_denoising(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a prediction task where the source is the original notes and the target is comprehensively noisy notes.
    """
    noisy_notes = add_comprehensive_noise(notes, noise_level=0.1)
    return notes, noisy_notes


def velocity_denoising(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    noisy_notes = add_noise_to_notes(
        notes=notes,
        attribute="velocity",
        noise_level=0.1,
    )
    return notes, noisy_notes


def pitch_denoising(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    noisy_notes = add_noise_to_notes(
        notes=notes,
        attribute="pitch",
        noise_level=0.1,
    )
    return notes, noisy_notes


def start_time_denoising(notes: pd.DataFrame) -> tuple[pd.DataFrame]:
    noisy_notes = add_noise_to_notes(
        notes=notes,
        attribute="start",
        noise_level=0.1,
    )
    return notes, noisy_notes


def time_denoising(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    noisy_notes = add_noise_to_notes(
        notes=notes,
        attribute="time",
        noise_level=0.1,
    )
    return notes, noisy_notes


def high_median_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    median = notes.pitch.median()
    source_notes = notes[notes.pitch < median]
    target_notes = notes[notes.pitch >= median]
    return source_notes, target_notes


def above_low_quartile_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.pitch.quantile(0.25)
    source_notes = notes[notes.pitch < q1]
    target_notes = notes[notes.pitch >= q1]
    return source_notes, target_notes


def above_high_quartile_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q3 = notes.pitch.quantile(0.75)
    source_notes = notes[notes.pitch < q3]
    target_notes = notes[notes.pitch >= q3]
    return source_notes, target_notes


def below_low_quartile_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.pitch.quantile(0.25)
    source_notes = notes[notes.pitch >= q1]
    target_notes = notes[notes.pitch < q1]
    return source_notes, target_notes


def below_high_quartile_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q3 = notes.pitch.quantile(0.75)
    source_notes = notes[notes.pitch >= q3]
    target_notes = notes[notes.pitch < q3]
    return source_notes, target_notes


def low_median_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    median = notes.pitch.median()
    source_notes = notes[notes.pitch >= median]
    target_notes = notes[notes.pitch < median]
    return source_notes, target_notes


def middle_quartiles_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.pitch.quantile(0.25)
    q3 = notes.pitch.quantile(0.75)
    source_notes = notes[(notes.pitch < q1) | (notes.pitch >= q3)]
    target_notes = notes[(notes.pitch >= q1) & (notes.pitch < q3)]
    return source_notes, target_notes


def extreme_quartiles_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.pitch.quantile(0.25)
    q3 = notes.pitch.quantile(0.75)
    target_notes = notes[(notes.pitch < q1) | (notes.pitch >= q3)]
    source_notes = notes[(notes.pitch >= q1) & (notes.pitch < q3)]
    return source_notes, target_notes


def loud_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    median = notes.velocity.median()
    soft_notes = notes[notes.velocity < median]
    loud_notes = notes[notes.velocity >= median]
    return soft_notes, loud_notes


def very_soft_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.velocity.quantile(0.25)
    source_notes = notes[notes.velocity >= q1]
    target_notes = notes[notes.velocity < q1]
    return source_notes, target_notes


def very_loud_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q3 = notes.velocity.quantile(0.75)
    source_notes = notes[notes.velocity < q3]
    target_notes = notes[notes.velocity >= q3]
    return source_notes, target_notes


def soft_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    median = notes.velocity.median()
    source_notes = notes[notes.velocity < median]
    target_notes = notes[notes.velocity >= median]
    return source_notes, target_notes


def moderate_velocity_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.velocity.quantile(0.25)
    q3 = notes.velocity.quantile(0.75)
    source_notes = notes[(notes.velocity < q1) | (notes.velocity >= q3)]
    target_notes = notes[(notes.velocity >= q1) & (notes.velocity < q3)]
    return source_notes, target_notes


def extreme_velocity_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = notes.velocity.quantile(0.25)
    q3 = notes.velocity.quantile(0.75)
    target_notes = notes[(notes.velocity < q1) | (notes.velocity >= q3)]
    source_notes = notes[(notes.velocity >= q1) & (notes.velocity < q3)]
    return source_notes, target_notes


prediction_task_to_token_pair = {
    # Two outdated tasks
    "bass_prediction": ("<BASS>", "<NO_BASS>"),
    "reverse_bass_prediction": ("<NO_BASS>", "<BASS>"),
    # One task with outdated name
    "high_median_prediction": ("<HIGH_FROM_MEDIAN>", "<LOW_FROM_MEDIAN>"),
    # Dynamically calculated pitch tasks for PIANO dataset
    "above_median_prediction": ("<HIGH_FROM_MEDIAN>", "<LOW_FROM_MEDIAN>"),
    "below_median_prediction": ("<LOW_FROM_MEDIAN>", "<HIGH_FROM_MEDIAN>"),
    "above_low_quartile_prediction": ("<ABOVE_LOW_QUARTILE>", "<BELOW_LOW_QUARTILE>"),
    "above_high_quartile_prediction": ("<ABOVE_HIGH_QUARTILE>", "<BELOW_HIGH_QUARTILE>"),
    "below_low_quartile_prediction": ("<BELOW_LOW_QUARTILE>", "<ABOVE_LOW_QUARTILE>"),
    "below_high_quartile_prediction": ("<BELOW_HIGH_QUARTILE>", "<ABOVE_HIGH_QUARTILE>"),
    "middle_quartiles_prediction": ("<MIDDLE_QUARTILES>", "<EXTREME_QUARTILES>"),
    "extreme_quartiles_prediction": ("<EXTREME_QUARTILES>", "<MIDDLE_QUARTILES>"),
    # Velocity tasks
    "loud_prediction": ("<LOUD>", "<SOFT>"),
    "very_soft_prediction": ("<ABOVE_VERY_SOFT>", "<VERY_SOFT>"),
    "very_loud_prediction": ("<VERY_LOUD>", "<BELOW_VERY_LOUD>"),
    "soft_prediction": ("<SOFT>", "<LOUD>"),
    "moderate_velocity_prediction": ("<MODERATE_VOLUME>", "<EXTREME_VOLUME>"),
    "extreme_velocity_prediction": ("<EXTREME_VOLUME>", "<MODERATE_VOLUME>"),
    # Denoising tasks
    "velocity_denoising": ("<CLEAN>", "<NOISY_VELOCITY>"),
    "pitch_denoising": ("<CLEAN>", "<NOISY_PITCH>"),
    "start_time_denoising": ("<CLEAN>", "<NOISY_START_TIME>"),
    "time_denoising": ("<CLEAN>", "<NOISY_TIME>"),
    "comprehensive_denoising": ("<CLEAN>", "<NOISY>"),
}


all_tasks = [
    # Dynamically calculated pitch tasks
    "high_median_prediction",
    "above_median_prediction",
    "low_median_prediction",
    "above_low_quartile_prediction",
    "above_high_quartile_prediction",
    "below_low_quartile_prediction",
    "below_high_quartile_prediction",
    "middle_quartiles_prediction",
    "extreme_quartiles_prediction",
    # Velocity tasks
    "loud_prediction",
    "very_soft_prediction",
    "very_loud_prediction",
    "soft_prediction",
    "moderate_velocity_prediction",
    "extreme_velocity_precition",
    "velocity_denoising",
    "pitch_denoising",
    "start_time_denoising",
    "time_denoising",
    "comprehensive_denoising",
]


task_generators = {
    "above_median_prediction": high_median_prediction,
    "above_low_quartile_prediction": above_low_quartile_prediction,
    "above_high_quartile_prediction": above_high_quartile_prediction,
    "below_low_quartile_prediction": below_low_quartile_prediction,
    "below_high_quartile_prediction": below_high_quartile_prediction,
    "below_median_prediction": low_median_prediction,
    "middle_quartiles_prediction": middle_quartiles_prediction,
    "extreme_quartiles_prediction": extreme_quartiles_prediction,
    "loud_prediction": loud_prediction,
    "very_soft_prediction": very_soft_prediction,
    "very_loud_prediction": very_loud_prediction,
    "soft_prediction": soft_prediction,
    "moderate_velocity_prediction": moderate_velocity_prediction,
    "extreme_velocity_prediction": extreme_velocity_prediction,
    "velocity_denoising": velocity_denoising,
    "pitch_denoising": pitch_denoising,
    "start_time_denoising": start_time_denoising,
    "time_denoising": time_denoising,
    "comprehensive_denoising": comprehensive_denoising,
}


def get_task_generator(task: str) -> callable:
    return task_generators.get(task)


def get_source_task_token(task: str) -> str:
    return prediction_task_to_token_pair[task][0]


def get_target_task_token(task: str) -> str:
    return prediction_task_to_token_pair[task][1]
