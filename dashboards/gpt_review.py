import os
import json
from glob import glob
from contextlib import nullcontext

import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf
from datasets import load_dataset
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from gpt2.model import GPT, GPTConfig
from dashboards.common.components import download_button
from tokenized_midi_datasets import OneTimeTokenDataset, AwesomeTokensDataset, ExponentialTimeTokenDataset


def main():
    with st.sidebar:
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox(label="device", options=devices)
        checkpoints = glob("checkpoints/*/*.pt")
        checkpoint_path = st.selectbox(label="checkpoint", options=checkpoints)

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

        checkpoint = torch.load(f=checkpoint_path, map_location=device)

        train_config = checkpoint["config"]
        cfg = OmegaConf.create(train_config)
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    base_dataset_path = st.text_input(label="base dataset", value="roszcz/maestro-sustain-v2")
    dataset_split = st.selectbox(label="split", options=["validation", "train", "test"])

    # This if is for compatibility with models trained before 1.05
    if "dataset_name" in train_config["data"].keys():
        config_name = cfg.data.dataset_name
        if cfg.data.tokenizer == "OneTimeTokenizer":
            dataset_name = "OneTimeTokenDataset"
            dataset_config = OneTimeTokenDataset.builder_configs[config_name].builder_parameters
            dataset_config = OmegaConf.create(dataset_config)

            tokenizer = OneTimeTokenizer(**dataset_config.tokenizer_parameters)

        elif cfg.data.tokenizer == "NoLossTokenizer":
            dataset_name = "ExponentialTimeTokenDataset"
            dataset_config = ExponentialTimeTokenDataset.builder_configs[config_name].builder_parameters
            dataset_config = OmegaConf.create(dataset_config)

            tokenizer = NoLossTokenizer(**dataset_config.tokenizer_parameters)

        elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
            dataset_name = "AwesomeTokensDataset"
            dataset_config = AwesomeTokensDataset.builder_configs[config_name].builder_parameters
            dataset_config = OmegaConf.create(dataset_config)

            min_time_unit = dataset_config.tokenizer_parameters["min_time_unit"]
            n_velocity_bins = dataset_config.tokenizer_parameters["n_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)

    else:
        dataset_config = cfg.dataset

        if cfg.data.tokenizer == "OneTimeTokenizer":
            dataset_name = "OneTimeTokenDataset"
            tokenizer = OneTimeTokenizer(**dataset_config.tokenizer_parameters)

        elif cfg.data.tokenizer == "NoLossTokenizer":
            dataset_name = "ExponentialTimeTokenDataset"
            tokenizer = NoLossTokenizer(**dataset_config.tokenizer_parameters)

        elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
            tokenizer_path = "pretrained/awesome_tokenizers/awesome-tokenizer-pretrained.json"
            min_time_unit = dataset_config.tokenizer_parameters["min_time_unit"]
            n_velocity_bins = dataset_config.tokenizer_parameters["n_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)
            dataset_name = "AwesomeTokensDataset"

    dataset_config.base_dataset_name = base_dataset_path
    # Do not use any extra datasets
    dataset_config.extra_datasets = []
    # Disable augmentation
    dataset_config.augmentation_repetitions = 0

    dataset = load_dataset(
        f"tokenized_midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **dataset_config,
    )

    # model init
    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=dataset_config.sequence_length,
        bias=cfg.model.bias,
        vocab_size=None,
        dropout=cfg.model.dropout,
    )  # start with model_args from command line

    checkpoint_model_args = checkpoint["model_args"]

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    with st.expander("train config"):
        st.json(train_config)

    with st.expander("dataset config"):
        st.write(dataset_config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]
    source = json.loads(record["source"])

    with st.expander(label="source"):
        st.json(source)

    notes = tokenizer.decode(record["note_token_ids"])
    piece = ff.MidiPiece(notes, source=source)
    st.write(
        """
        If the tokenizer sees unmatched NOTE_ON events
        it will treat them as if the note was pressed until the end of the recording.
        If the tokenizef sees umatched NOTE_OFF rents, it will ignore it.
        If it encounters a NaN or end <= start, the note is invalid.
        """
    )
    temperature = st.number_input(label="temperature", value=1.0)
    max_new_tokens = st.number_input(label="max_new_tokens", value=dataset_config.sequence_length)

    input_sequence = torch.tensor([record["note_token_ids"]], device=device)
    with torch.no_grad():
        with ctx:
            output = model.generate(
                idx=input_sequence,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    output: torch.Tensor = output[0].cpu().numpy()  # , dataset_config.sequence_length :]
    # we want to decode the whole output so that the pressed notes can be unpressed by the tokenizer
    out_notes = tokenizer.decode(output)
    out_piece = ff.MidiPiece(out_notes)
    # start from new model-generated notes
    generated_notes = out_notes.iloc[piece.size :].copy()
    generated_piece = ff.MidiPiece(df=generated_notes)
    generated_piece.time_shift(-generated_piece.df.start.min())
    io_columns = st.columns(2)

    title = source["title"]
    composer = source["composer"]
    piece_name = (title + composer).replace(" ", "_").casefold()

    with io_columns[0]:
        st.write("original:")
        streamlit_pianoroll.from_fortepyan(piece=piece)

        with st.expander(label="tokens"):
            st.write([tokenizer.vocab[idx] for idx in record["note_token_ids"]])

        original_midi_path = f"tmp/fragment_of_{piece_name}.mid"
        source_file = piece.to_midi()

        try:
            source_file.write(original_midi_path)
            with open(original_midi_path, "rb") as file:
                download_button_str = download_button(
                    object_to_download=file.read(),
                    download_filename=original_midi_path.split("/")[-1],
                    button_text="Download source midi",
                )
                st.markdown(download_button_str, unsafe_allow_html=True)
        finally:
            # make sure to always clean up
            os.unlink(original_midi_path)

    with io_columns[1]:
        st.write("generated:")

        streamlit_pianoroll.from_fortepyan(piece=generated_piece)

        output = output[input_sequence.shape[-1] :]
        with st.expander("generated tokens"):
            st.write([tokenizer.vocab[idx] for idx in output])

        milion_parameters = model.get_num_params() / 1e6
        midi_path = f"tmp/{milion_parameters:.0f}_variations_on_{piece_name}.mid"
        generated_file = generated_piece.to_midi()

        try:
            generated_file.write(midi_path)
            with open(midi_path, "rb") as file:
                download_button_str = download_button(
                    object_to_download=file.read(),
                    download_filename=midi_path.split("/")[-1],
                    button_text="Download generated midi",
                )
                st.markdown(download_button_str, unsafe_allow_html=True)
        finally:
            # make sure to always clean up
            os.unlink(midi_path)

    st.write("whole model output")
    generated_notes_with_offset = out_notes[piece.size :].copy()
    second_part = ff.MidiPiece(generated_notes_with_offset)

    # Model could have also add "NOTE_OFF" events to original sequence
    expanded_input_notes = out_notes[: piece.size].copy()
    expanded_piece = ff.MidiPiece(expanded_input_notes)
    streamlit_pianoroll.from_fortepyan(piece=expanded_piece, secondary_piece=second_part)

    full_midi_path = f"tmp/full_{milion_parameters}_variations_on_{piece_name}.mid"
    out_file = out_piece.to_midi()
    try:
        out_file.write(full_midi_path)
        with open(full_midi_path, "rb") as file:
            download_button_str = download_button(
                object_to_download=file.read(),
                download_filename=full_midi_path.split("/")[-1],
                button_text="Download midi with context",
            )
            st.markdown(download_button_str, unsafe_allow_html=True)
    finally:
        # make sure to always clean up
        os.unlink(full_midi_path)


if __name__ == "__main__":
    main()
