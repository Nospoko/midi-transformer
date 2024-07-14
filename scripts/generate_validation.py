from contextlib import AbstractContextManager

import torch
from omegaconf import DictConfig

from gpt2.model import GPT
import data.database_manager as database_manager
from gpt2.generation import generate_from_validation_example
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


def prepare_validation_examples_for_task(cfg: DictConfig) -> list[dict]:
    validation_examples = database_manager.get_validation_examples_for_task(task=cfg["task"])
    prepared_examles = []

    def process_row(row):
        example = {
            "generation_parameters": row[database_manager.parameter_dtype.keys()].to_dict(),
            "prompt": row[database_manager.prompt_dtype.keys()].to_dict(),
        }
        prepared_examles.append(example)

    validation_examples.apply(process_row, axis=1)

    return prepared_examles


def run_generation_step(
    model: GPT,
    checkpoint: dict,
    run_name: str,
    validation_examples: list[dict],
    tokenizer: AwesomeTokenizer | ExponentialTokenizer,
    device: torch.device,
    ctx: AbstractContextManager,
):
    _, model_id = database_manager.register_model_from_checkpoint(
        checkpoint=checkpoint,
        run_name=run_name,
    )
    generations = []
    for example in validation_examples:
        generated_notes = generate_from_validation_example(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            parameters=example["generation_parameters"],
            device=device,
            ctx=ctx,
        )

        generated_info = {
            "parameters_id": example["generation_parameters"]["parameters_id"],
            "prompt_id": example["prompt"]["prompt_id"],
            "model_id": model_id,
            "generated_notes": generated_notes.to_json(),
        }

        generations.append(generated_info)

    database_manager.insert_validation_generations_batch(generations=generations)
    print(f"Populated dataset with {len(validation_examples)} generations!")
