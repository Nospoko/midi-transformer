from contextlib import AbstractContextManager

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from artifacts import special_tokens
from gpt2.model import GPT, GPTConfig
import data.database_manager as database_manager
from gpt2.generation import generate_from_validation_example
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


def prepare_validation_examples_for_task(cfg: DictConfig) -> list[dict]:
    prepared_examles = []

    def process_row(row):
        example = {
            "generation_parameters": row[database_manager.parameter_dtype.keys()].to_dict(),
            "prompt": row[database_manager.prompt_dtype.keys()].to_dict(),
        }
        prepared_examles.append(example)

    if cfg["task"] == "multi":
        for task in cfg["tasks"]:
            validation_examples = database_manager.get_validation_examples_for_task(task=task)
            validation_examples.apply(process_row, axis=1)
        return

    validation_examples = database_manager.get_validation_examples_for_task(task=cfg["task"])
    prepared_examles = []
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
    model_config=None,
):
    if validation_examples is None:
        return
    _, model_id = database_manager.register_model_from_checkpoint(
        checkpoint=checkpoint,
        run_name=run_name,
    )
    for example in validation_examples:
        generated_notes, tokenized_prompt = generate_from_validation_example(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            parameters=example["generation_parameters"],
            device=device,
            ctx=ctx,
            model_config=model_config,
        )

        generated_info = {
            "parameters_id": example["generation_parameters"]["parameters_id"],
            "prompt_id": example["prompt"]["prompt_id"],
            "model_id": model_id,
            "tokenized_prompt": tokenized_prompt.to_json(),
            "generated_notes": generated_notes.to_json(),
        }
        database_manager.insert_validation_generations_batch(generations=[generated_info])
    print(f"Populated dataset with {len(validation_examples)} generations!")


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(cfg: DictConfig):
    if "tokenizer" in cfg:
        tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.tokenizer_parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}

        if cfg.tokenizer.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = to_absolute_path(
                f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            )
            return AwesomeTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTokenizer(**tokenizer_parameters)
    else:
        tokenizer_parameters = OmegaConf.to_container(cfg.data.tokenizer_parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}

        if cfg.data.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = to_absolute_path(
                f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            )
            return AwesomeTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTokenizer(**tokenizer_parameters)


def initialize_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
    pad_token_id: int = 0,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (DictConfig): The configuration object.
        dataset_config (dict): The dataset configuration.
        checkpoint (dict): The model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        GPT: The initialized GPT model.
    """
    model_args = {
        "n_layer": cfg.model.n_layer,
        "n_head": cfg.model.n_head,
        "n_embd": cfg.model.n_embd,
        "block_size": cfg.data.sequence_length,
        "bias": cfg.model.bias,
        "vocab_size": None,
        "dropout": cfg.model.dropout,
    }

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_token_id=pad_token_id)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model
