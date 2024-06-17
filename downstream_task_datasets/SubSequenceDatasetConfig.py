import datasets
from datasets import BuilderConfig

from data.subsequence_dataset import special_tokens


class SubSequenceDatasetConfig(BuilderConfig):
    """
    Configuration class for creating a sub-sequence MIDI dataset.

    Attributes:
        base_dataset_name (str): Name of the base dataset.
        extra_datasets (list[str]): List of additional datasets.
        sequence_length (int): Length of the sequences.
        sequence_step (int): Step size between sequences.
        pause_detection_threshold (int): Threshold for detecting pauses.
        augmentation (dict): Parameters for augmentation
    """

    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-sustain-v2",
        extra_datasets: list[str] = [],
        sequence_length: int = 64,
        sequence_step: int = 42,
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
            sequence_length (int): Length of the sequences.
            sequence_step (int): Step size between sequences.
            pause_detection_threshold (int): Threshold for detecting pauses.
            augmentation (dict): Parameters for augmentation (max_pitch_shift, speed_change_factors))
            **kwargs: Additional keyword arguments.
        """
        # Initialize the version and other parameters
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        # Assign the provided arguments to the class attributes
        self.base_dataset_name: str = base_dataset_name
        self.extra_datasets: list[str] = extra_datasets
        self.sequence_length: int = sequence_length
        self.sequence_step: int = sequence_step
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
            "sequence_length": self.sequence_length,
            "sequence_step": self.sequence_step,
            "pause_detection_threshold": self.pause_detection_threshold,
            "augmentation": self.augmentation,
        }


# Define coarse tokenizer parameters for future pre-training on coarse datasets
coarse_tokenizer_parameters = {"min_time_unit": 0.01, "n_velocity_bins": 32, "special_tokens": special_tokens}
augmentation_parameters = {
    "speed_change_factors": [0.95, 0.975, 1.05, 1.025],
    "max_pitch_shift": 5,
}
# List of configurations for different datasets for debugging
BUILDER_CONFIGS = [
    SubSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        name="basic-no-overlap",
    ),
    SubSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        name="giant-no-overlap",
    ),
    # Non-overlapping coarse datasets with augmentation
    SubSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        augmentation=augmentation_parameters,
        name="basic-no-overlap-augmented",
    ),
    SubSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        augmentation=augmentation_parameters,
        name="giant-no-overlap-augmented",
    ),
    # Colossal datasets
    SubSequenceDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2", "roszcz/pianofor-ai-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        name="colossal-no-overlap",
    ),
]
