import pandas as pd


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


def middle_quantiles_prediction(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


prediction_task_to_token_pair = {
    # Two outdated tasks
    "bass_prediction": ("<BASS>", "<NO_BASS>"),
    "reverse_bass_prediction": ("<NO_BASS>", "<BASS>"),
    # Dynamically calculated pitch tasks
    "high_median_prediction": ("<HIGH_FROM_MEDIAN>", "<LOW_FROM_MEDIAN>"),
    "low_median_prediction": ("<LOW_FROM_MEDIAN>", "<HIGH_FROM_MEDIAN>"),
    "above_low_quartile_prediction": ("<ABOVE_LOW_QUARTILE>", "<BELOW_LOW_QUARTILE>"),
    "above_high_quartile_prediction": ("<ABOVE_HIGH_QUARTILE>", "<BELOW_HIGH_QUARTILE>"),
    "below_low_quartile_prediction": ("<BELOW_LOW_QUARTILE>", "<ABOVE_LOW_QUARTILE>"),
    "below_high_quartile_prediction": ("<BELOW_HIGH_QUARTILE", "<ABOVE_HIGH_QUARTILE>"),
    "middle_quartiles_prediction": ("<MIDDLE_QUARTILES>", "<EXTREME_QUARTILES>"),
    "extreme_quartiles_prediction": ("<EXTREME_QUARTILES>", "<MIDDLE_QUARTILES>"),
    # Velocity tasks
    "loud_prediction": ("<LOUD>", "<SOFT>"),
    "very_soft_prediction": ("<ABOVE_VERY_SOFT>", "<VERY_SOFT>"),
    "very_loud_prediction": ("<VERY_LOUD>", "<BELOW_VERY_LOUD>"),
    "soft_prediction": ("<SOFT>", "<LOUD>"),
    "moderate_velocity_prediction": ("<MODERATE_VELOCITY>", "<EXTREME_VELOCITY>"),
}
all_tasks = [
    # Dynamically calculated pitch tasks
    "high_median_prediction",
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
]


task_generators = {
    "high_median_prediction": high_median_prediction,
    "above_low_quartile_prediction": above_low_quartile_prediction,
    "above_high_quartile_prediction": above_high_quartile_prediction,
    "below_low_quartile_prediction": below_low_quartile_prediction,
    "below_high_quartile_prediction": below_high_quartile_prediction,
    "low_median_prediction": low_median_prediction,
    "middle_quantiles_prediction": middle_quantiles_prediction,
    "extreme_quantiles_prediction": extreme_quartiles_prediction,
    "loud_prediction": loud_prediction,
    "very_soft_prediction": very_soft_prediction,
    "very_loud_prediction": very_loud_prediction,
    "soft_prediction": soft_prediction,
    "moderate_velocity_prediction": moderate_velocity_prediction,
}


def get_task_generator(task: str) -> callable:
    return task_generators.get(task)


def get_source_task_token(task: str) -> str:
    return prediction_task_to_token_pair[task][0]


def get_target_task_token(task: str) -> str:
    return prediction_task_to_token_pair[task][1]
