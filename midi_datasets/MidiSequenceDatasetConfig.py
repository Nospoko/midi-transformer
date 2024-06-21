import datasets
from datasets import BuilderConfig


class MidiSequenceDatasetConfig(BuilderConfig):
    """
    Configuration class for creating a sub-sequence MIDI dataset.

    Attributes:
        base_dataset_name (str): Name of the base dataset.
        extra_datasets (list[str]): List of additional datasets.
        notes_per_record (int): Length of the sequences.
        step (int): Step size between sequences.
        pause_detection_threshold (int): Threshold for detecting pauses.
        augmentation (dict): Parameters for augmentation
    """

    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-sustain-v2",
        extra_datasets: list[str] = [],
        notes_per_record: int = 64,
        step: int = 42,
        pause_detection_threshold: int = 4,
        augmentation: dict = {
            "speed_change_factors": None,
            "max_pitch_shift": 0,
        },
        **kwargs,
    ):
        """
        Initialize the SubSequenceDatasetConfig.

        Parameters:
            base_dataset_name (str): Name of the base dataset.
            extra_datasets (list[str]): List of additional datasets.
            notes_per_record (int): Length of the sequences.
            step (int): Step size between sequences.
            pause_detection_threshold (int): Threshold for detecting pauses.
            augmentation (dict): Parameters for augmentation (max_pitch_shift, speed_change_factors))
            **kwargs: Additional keyword arguments.
        """
        # Initialize the version and other parameters
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        # Assign the provided arguments to the class attributes
        self.base_dataset_name: str = base_dataset_name
        self.extra_datasets: list[str] = extra_datasets
        self.notes_per_record: int = notes_per_record
        self.step: int = step
        self.pause_detection_threshold = pause_detection_threshold
        self.augmentation = augmentation

    @property
    def builder_parameters(self):
        """
        Returns the builder parameters as a dictionary.

        Returns:
            dict: Builder parameters.
        """
        return {
            "base_dataset_name": self.base_dataset_name,
            "extra_datasets": self.extra_datasets,
            "notes_per_record": self.notes_per_record,
            "step": self.step,
            "pause_detection_threshold": self.pause_detection_threshold,
            "augmentation": self.augmentation,
        }


augmentation_parameters = {
    "speed_change_factors": [0.95, 0.975, 1.05, 1.025],
    "max_pitch_shift": 5,
}
# List of configurations for different datasets for debugging
BUILDER_CONFIGS = [
    MidiSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        notes_per_record=512,
        step=512,
        pause_detection_threshold=4,
        name="basic-no-overlap",
    ),
    MidiSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        notes_per_record=512,
        step=512,
        pause_detection_threshold=4,
        name="giant-no-overlap",
    ),
    # Non-overlapping coarse datasets with augmentation
    MidiSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        notes_per_record=512,
        step=512,
        pause_detection_threshold=4,
        augmentation=augmentation_parameters,
        name="basic-no-overlap-augmented",
    ),
    MidiSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        notes_per_record=512,
        step=512,
        pause_detection_threshold=4,
        augmentation=augmentation_parameters,
        name="giant-no-overlap-augmented",
    ),
    # Colossal datasets
    MidiSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2", "roszcz/pianofor-ai-sustain-v2"],
        notes_per_record=512,
        step=512,
        pause_detection_threshold=4,
        name="colossal-no-overlap",
    ),
]
