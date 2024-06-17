import datasets
from datasets import BuilderConfig

from data.subsequence_dataset import special_tokens


class TokenizedMidiDatasetConfig(BuilderConfig):
    """
    Configuration class for creating a tokenized MIDI dataset.

    Attributes:
        base_dataset_name (str): Name of the base dataset.
        extra_datasets (list[str]): List of additional datasets.
        sequence_length (int): Length of the sequences.
        sequence_step (int): Step size between sequences.
        pause_detection_threshold (int): Threshold for detecting pauses.
        tokenizer_parameters (dict): Parameters for the tokenizer.
        augmentation (dict): Parameters for augmentation
    """

    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-sustain-v2",
        extra_datasets: list[str] = [],
        sequence_length: int = 64,
        sequence_step: int = 42,
        pause_detection_threshold: int = 4,
        tokenizer_parameters: dict = {"min_time_unit": 0.01, "n_velocity_bins": 32, "special_tokens": special_tokens},
        augmentation: dict = {
            "speed_change_factors": None,
            "max_pitch_shift": 0,
        },
        **kwargs,
    ):
        """
        Initialize the TokenizedMidiDatasetConfig.

        Parameters:
            base_dataset_name (str): Name of the base dataset.
            extra_datasets (list[str]): List of additional datasets.
            sequence_length (int): Length of the sequences.
            sequence_step (int): Step size between sequences.
            pause_detection_threshold (int): Threshold for detecting pauses.
            tokenizer_parameters (dict): Parameters for the tokenizer.
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
        self.tokenizer_parameters = tokenizer_parameters
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
            "tokenizer_parameters": self.tokenizer_parameters,
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
    # Default
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters={"min_time_unit": 0.01, "n_velocity_bins": 32},
        name="basic",
    ),
    # Non-overlapping coarse datasets
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="basic-no-overlap",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="giant-no-overlap",
    ),
    # Non-overlapping coarse datasets with augmentation
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation=augmentation_parameters,
        name="basic-no-overlap-augmented",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation=augmentation_parameters,
        name="giant-no-overlap-augmented",
    ),
    # Colossal datasets
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2", "roszcz/pianofor-ai-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="colossal-no-overlap",
    ),
]
