import time
from typing import List

import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset as HuggingFaceDataset

from artifacts import special_tokens
from data.piano_dataset import PianoDataset
from data.tokenizer import ExponentialTokenizer


def test_piano_dataset_performance(
    dataset: HuggingFaceDataset,
    tokenizer: ExponentialTokenizer,
    sequence_length: int,
    notes_per_record: int,
    tasks: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
) -> pd.DataFrame:
    results = []
    for task in tasks:
        piano_dataset = PianoDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            notes_per_record=notes_per_record,
            tasks=[task],
            loss_masking="finetuning",
        )
        print(f"task: {task}, dataset length: {len(piano_dataset)}")

        dataloader = DataLoader(
            piano_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        start_time = time.time()
        it = 0
        for _ in dataloader:
            it += 1
            if it % 100 == 1:
                current_time = time.time()
                elapsed = current_time - start_time
                iterations_per_second = it / elapsed
                current_result = {
                    "task": task,
                    "iterations_per_second": iterations_per_second,
                    "tokens_per_second": iterations_per_second * sequence_length * batch_size,
                }
                print(current_result)
        end_time = time.time()

        iteration_time = end_time - start_time
        iterations_per_second = len(piano_dataset) / iteration_time
        result = {
            "task": task,
            "total_time": iteration_time,
            "iterations_per_second": iterations_per_second,
            "tokens_per_second": iterations_per_second * sequence_length,
        }
        print(result)
        results.append(result)

    return pd.DataFrame(results)


def main():
    # Create a dummy dataset and tokenizer
    tokenizer = ExponentialTokenizer(min_time_unit=0.01, n_velocity_bins=32, special_tokens=special_tokens)

    # Set up test parameters
    sequence_length = 1024
    notes_per_record = 128
    tasks = [
        "above_median_prediction",
        "below_median_prediction",
        "above_low_quartile_prediction",
        "above_high_quartile_prediction",
        "below_low_quartile_prediction",
        "below_high_quartile_prediction",
        "middle_quartiles_prediction",
        "extreme_quartiles_prediction",
        "loud_prediction",
        "very_soft_prediction",
        "very_loud_prediction",
        "soft_prediction",
        "moderate_velocity_prediction",
        "extreme_velocity_prediction",
        "velocity_denoising",
        "pitch_denoising",
        "start_time_denoising",
        "time_denoising",
        "comprehensive_denoising",
    ]

    dataset_config = {
        "augmentation": {
            "max_pitch_shift": 0,
            "speed_change_factors": [],
        },
        "base_dataset_name": "roszcz/maestro-sustain-v2",
        "extra_datasets": [],
        "pause_detection_threshold": 4,
    }
    dataset_path = "./midi_datasets/AugmentedDataset"

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=8,
        **dataset_config,
    )
    dataset = dataset["train"]
    dataset = dataset.shard(1024, 512)
    # Run the performance test
    results = test_piano_dataset_performance(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        notes_per_record=notes_per_record,
        tasks=tasks,
        batch_size=32,
        num_workers=8,  # Set to 0 for single-threaded operation, increase for multi-threading
    )

    # Print and analyze results
    print(results.sort_values("iterations_per_second", ascending=False))
    print(f"Average iterations per second: {results['iterations_per_second'].mean():.2f}")
    print(f"Average tokens per second: {results['tokens_per_second'].mean():.2f}")
    print(f"Total time for all tasks: {results['total_time'].sum():.2f} seconds")


if __name__ == "__main__":
    main()
