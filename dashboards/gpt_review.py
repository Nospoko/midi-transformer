import os
import json
from glob import glob
from typing import Tuple
from contextlib import nullcontext

import yaml
import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf
from datasets import Dataset, load_dataset
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer

from gpt2.model import GPT, GPTConfig
from dashboards.common.components import download_button


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
    device: torch.device,
) -> Tuple[OmegaConf, dict, AwesomeMidiTokenizer | ExponentialTimeTokenizer]:
    """
    Loads the model configuration, dataset configuration, and tokenizer based on the checkpoint path.

    Parameters:
        checkpoint (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        Tuple[OmegaConf, dict, object]: The configuration, dataset configuration, dataset name, and tokenizer.
    """
    train_config = checkpoint["config"]
    cfg = OmegaConf.create(train_config)
    dataset_config = cfg.dataset
    if cfg.data.tokenizer == "OneTimeTokenizer":
        tokenizer = OneTimeTokenizer(**dataset_config["tokenizer_parameters"])
    # NoLossTokenizer for backward-compatibility
    elif cfg.data.tokenizer == "ExponentialTimeTokenizer" or cfg.data.tokenizer == "NoLossTokenizer":
        tokenizer = ExponentialTimeTokenizer(**dataset_config["tokenizer_parameters"])
    elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
        min_time_unit = dataset_config["tokenizer_parameters"]["min_time_unit"]
        n_velocity_bins = dataset_config["tokenizer_parameters"]["n_velocity_bins"]
        tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
        tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)

    return cfg, dataset_config, tokenizer


def initialize_model(
    cfg: OmegaConf,
    dataset_config: dict,
    checkpoint: dict,
    device: torch.device,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (OmegaConf): The configuration object.
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
        "block_size": dataset_config["sequence_length"],
        "bias": cfg.model.bias,
        "vocab_size": None,
        "dropout": cfg.model.dropout,
    }

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def main():
    with st.sidebar:
        # Select device and checkpoint path
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox("device", options=devices)
        checkpoint_path = st.selectbox("checkpoint", options=glob("checkpoints/*/*.pt"))

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = "cuda" if "cuda" in device else "cpu"
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(
                device_type=device_type,
                dtype=torch.float32,
            )
        )

        # Load model, tokenizer, and configurations
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg, dataset_config, tokenizer = load_tokenizer(checkpoint, device)
        model = initialize_model(cfg, dataset_config, checkpoint=checkpoint, device=device)

    base_dataset_path = st.text_input("base dataset", value="roszcz/maestro-sustain-v2")
    dataset_split = st.selectbox("split", options=["validation", "train", "test"])

    dataset_config["base_dataset_name"] = base_dataset_path
    dataset_config["extra_datasets"] = []
    dataset_config["augmentation_repetitions"] = 0

    dataset = load_dataset(
        "tokenized_midi_datasets/MidiSequenceDataset",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **dataset_config,
    )

    # Select part of the dataset
    dataset = select_part_dataset(midi_dataset=dataset)

    # Get the record id from user input
    idx = st.number_input("record_id", value=0, max_value=len(dataset))
    record = dataset[idx]
    source = json.loads(record["source"])

    # Decode and display the original piece
    notes = pd.DataFrame(record["notes"])
    piece = ff.MidiPiece(notes, source=source)

    with st.form("generate parameters"):
        temperature = st.number_input("temperature", value=1.0)
        max_new_tokens = st.number_input("max_new_tokens", value=dataset_config["sequence_length"])
        run = st.form_submit_button("Generate")

    if not run:
        return

    # Generate new tokens and create the generated piece
    note_token_ids = tokenizer.encode(notes)
    input_sequence = torch.tensor(note_token_ids, device=device)
    with torch.no_grad():
        with ctx:
            output = model.generate(input_sequence, max_new_tokens=max_new_tokens, temperature=temperature)

    output = output[0].cpu().numpy()
    out_notes = tokenizer.decode(output)
    out_piece = ff.MidiPiece(out_notes)
    generated_notes = out_notes.iloc[piece.size :].copy()
    generated_piece = ff.MidiPiece(generated_notes)
    generated_piece.time_shift(-generated_piece.df.start.min())

    io_columns = st.columns(2)
    title, composer = source["title"], source["composer"]
    piece_name = (title + composer).replace(" ", "_").casefold()

    # Display and allow download of the original MIDI
    with io_columns[0]:
        st.write("original:")
        streamlit_pianoroll.from_fortepyan(piece=piece)
        original_midi_path = f"tmp/fragment_of_{piece_name}_{idx}.mid"
        source_file = piece.to_midi()
        source_file.write(original_midi_path)
        with open(original_midi_path, "rb") as file:
            st.markdown(
                download_button(file.read(), original_midi_path.split("/")[-1], "Download source midi"),
                unsafe_allow_html=True,
            )
        os.unlink(original_midi_path)

    # Display and allow download of the generated MIDI
    with io_columns[1]:
        st.write("generated:")
        streamlit_pianoroll.from_fortepyan(piece=generated_piece)
        output = output[input_sequence.shape[-1] :]
        milion_parameters = model.get_num_params() / 1e6
        midi_path = f"tmp/{milion_parameters:.0f}_variations_on_{piece_name}_{idx}.mid"
        generated_file = generated_piece.to_midi()
        generated_file.write(midi_path)
        with open(midi_path, "rb") as file:
            st.markdown(
                download_button(file.read(), midi_path.split("/")[-1], "Download generated midi"),
                unsafe_allow_html=True,
            )
        os.unlink(midi_path)

    st.write("whole model output")
    second_part = ff.MidiPiece(out_notes[piece.size :].copy())
    expanded_piece = ff.MidiPiece(out_notes[: piece.size].copy())
    streamlit_pianoroll.from_fortepyan(piece=expanded_piece, secondary_piece=second_part)

    # Allow download of the full MIDI with context
    full_midi_path = f"tmp/full_{milion_parameters}_variations_on_{piece_name}_{idx}.mid"
    out_piece.to_midi().write(full_midi_path)
    with open(full_midi_path, "rb") as file:
        st.markdown(
            download_button(file.read(), full_midi_path.split("/")[-1], "Download midi with context"),
            unsafe_allow_html=True,
        )
    os.unlink(full_midi_path)


if __name__ == "__main__":
    main()
