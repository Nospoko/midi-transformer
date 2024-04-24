from glob import glob
from contextlib import nullcontext

import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf
from datasets import load_dataset
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from model import GPT, GPTConfig
from tokenized_midi_datasets import OneTimeTokenDataset, ExponentialTimeTokenDataset


def main():
    with st.sidebar:
        device = st.selectbox(label="device", options=["cpu", "cuda:0"])
        checkpoints = glob("checkpoints/*.pt")
        checkpoint_path = st.selectbox(label="checkpoint", options=checkpoints)

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

        checkpoint = torch.load(f=checkpoint_path, map_location=device)
        model_config = checkpoint["config"]
        cfg = OmegaConf.create(model_config)
        config_name = cfg.data.dataset_name
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if cfg.data.tokenizer == "OneTimeTokenizer":
        dataset_name = "OneTimeTokenDataset"
        dataset_config = OneTimeTokenDataset.builder_configs[config_name]
        tokenizer = OneTimeTokenizer(**dataset_config.tokenizer_parameters)

    elif cfg.data.tokenizer == "NoLossTokenizer":
        dataset_name = "ExponentialTimeTokenDataset"
        dataset_config = ExponentialTimeTokenDataset.builder_configs[config_name]
        tokenizer = NoLossTokenizer(**dataset_config.tokenizer_parameters)
    dataset_split = st.selectbox(label="split", options=["validation", "train", "test"])

    dataset = load_dataset(
        f"tokenized_midi_datasets/{dataset_name}",
        name=config_name,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
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

    with st.expander("config"):
        st.write(dataset_config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]

    with st.expander(label="source"):
        st.json(record["source"])

    notes = tokenizer.decode(record["note_token_ids"])
    piece = ff.MidiPiece(notes, source=record["source"])
    st.write(
        """
        If the tokenizer sees unmatched NOTE_OFF or NOTE_ON events
        it will treat them as if the notes were playing on the edges of the recording.
        """
    )
    temperature = st.number_input(label="temperature", value=1.0)
    io_columns = st.columns(2)

    with io_columns[0]:
        st.write("original:")
        streamlit_pianoroll.from_fortepyan(piece=piece)

        with st.expander(label="tokens"):
            st.write([tokenizer.vocab[idx] for idx in record["note_token_ids"]])

    with io_columns[1]:
        st.write("generated:")

        input_sequence = torch.tensor([record["note_token_ids"]], device=device)
        with torch.no_grad():
            with ctx:
                output = model.generate(
                    idx=input_sequence, max_new_tokens=dataset_config.sequence_length, temperature=temperature
                )

        output = output[0, dataset_config.sequence_length :]
        generated_notes = tokenizer.decode(output)
        generated_piece = ff.MidiPiece(df=generated_notes)
        streamlit_pianoroll.from_fortepyan(piece=generated_piece)
        with st.expander("generated tokens"):
            st.write([tokenizer.vocab[idx] for idx in output])


if __name__ == "__main__":
    main()
