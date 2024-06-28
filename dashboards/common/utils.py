from typing import Tuple

import yaml
import torch
import streamlit as st
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig

from artifacts import special_tokens
from gpt2.model import GPT, GPTConfig
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


def select_part_dataset(midi_dataset: Dataset) -> Dataset:
    """
    Allows the user to select a part of the dataset based on composer and title.

    Parameters:
        midi_dataset (Dataset): The MIDI dataset to select from.

    Returns:
        Dataset: The selected part of the dataset.
    """
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        "Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox("Select title", options=piece_titles)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def load_tokenizer(
    checkpoint: str,
) -> Tuple[DictConfig, dict, AwesomeTokenizer | ExponentialTokenizer]:
    """
    Loads the model configuration, dataset configuration, and tokenizer based on the checkpoint path.

    Parameters:
        checkpoint (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        Tuple[DictConfig, dict, object]: The configuration, dataset configuration, dataset name, and tokenizer.
    """
    train_config = checkpoint["config"]
    cfg = OmegaConf.create(train_config)
    dataset_config = cfg.dataset

    if cfg.data.tokenizer == "ExponentialTimeTokenizer":
        tokenizer = ExponentialTokenizer(**cfg.data.tokenizer_parameters, special_tokens=special_tokens)
    elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
        min_time_unit = cfg.data.tokenizer_parameters["min_time_unit"]
        n_velocity_bins = cfg.data.tokenizer_parameters["n_velocity_bins"]
        tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
        tokenizer = AwesomeTokenizer.from_file(tokenizer_path)

    return cfg, dataset_config, tokenizer


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


@st.cache_data
def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


@st.cache_data
def load_hf_dataset(dataset_path: str, dataset_split: str):
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset
