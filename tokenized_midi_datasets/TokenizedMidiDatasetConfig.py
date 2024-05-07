import datasets
from datasets import BuilderConfig

from data.masked_midi_dataset import special_tokens


class TokenizedMidiDatasetConfig(BuilderConfig):
    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-v1-sustain",
        extra_datasets: list[str] = [],
        sequence_length: int = 64,
        sequence_step: int = 42,
        pause_detection_threshold: int = 4,
        tokenizer_parameters: dict = {"min_time_unit": 0.01, "n_velocity_bins": 32, "special_tokens": special_tokens},
        augmentation_probability: float = 0.0,
        augmentation_repetitions: int = 0,
        **kwargs,
    ):
        # Version history:
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        self.base_dataset_name: str = base_dataset_name
        self.extra_datasets: list[str] = extra_datasets
        self.sequence_length: int = sequence_length
        self.sequence_step: int = sequence_step
        self.tokenizer_parameters = tokenizer_parameters
        self.pause_detection_threshold = pause_detection_threshold
        self.augmentation_probability = augmentation_probability
        self.augmentation_repetitions = augmentation_repetitions

    @property
    def builder_parameters(self):
        return {
            "base_dataset_name": self.base_dataset_name,
            "extra_datasets": self.extra_datasets,
            "sequence_length": self.sequence_length,
            "sequence_step": self.sequence_step,
            "tokenizer_parameters": self.tokenizer_parameters,
            "pause_detection_threshold": self.pause_detection_threshold,
            "augmentation_probability": self.augmentation_probability,
            "augmentation_repetitions": self.augmentation_repetitions,
        }


# For the future pre-training I plan on using coarse datasets,
# so high-resolution tokenizers can remain the same -
# it will allow our first models to work properly on this branch.

coarse_tokenizer_parameters = {"min_time_unit": 0.01, "n_velocity_bins": 32, "special_tokens": special_tokens}
BUILDER_CONFIGS = [
    # High resolution datasets - no augmentation
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
        name="giant-mid",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
        name="basic-mid",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
        name="giant-long",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
        name="basic-long",
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
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="basic-no-overlap-augmented",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=512,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="giant-no-overlap-augmented",
    ),
    # Coarse datasets
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="giant-mid-coarse",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="basic-mid-coarse",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="giant-long-coarse",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        name="basic-long-coarse",
    ),
    # Augmented coarse datasets
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="giant-mid-coarse-augmented",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2"],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="giant-long-coarse-augmented",
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
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2", "roszcz/pianofor-ai-sustain-v2"],
        sequence_length=512,
        sequence_step=64,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="colossal-mid-coarse-augmented",
    ),
    TokenizedMidiDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=["roszcz/giant-midi-sustain-v2", "roszcz/pianofor-ai-sustain-v2"],
        sequence_length=1024,
        sequence_step=128,
        pause_detection_threshold=4,
        tokenizer_parameters=coarse_tokenizer_parameters,
        augmentation_probability=0.2,
        augmentation_repetitions=5,
        name="colossal-long-coarse-augmented",
    ),
]
