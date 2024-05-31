import json
from glob import glob
from contextlib import nullcontext

import torch
import numpy as np
import streamlit as st
from omegaconf import OmegaConf
from datasets import load_dataset
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer

from gpt2.model import GPT, GPTConfig
from data.next_token_dataset import NextTokenDataset


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
        config_name = cfg.data.dataset_name
        ptdtype = {"float32": torch.float32, "bfloat16": torch.float16, "float16": torch.float16}[cfg.system.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dataset_config = cfg.dataset
    if cfg.data.tokenizer == "OneTimeTokenizer":
        dataset_name = "OneTimeTokenDataset"
        tokenizer = OneTimeTokenizer(**dataset_config["tokenizer_parameters"])
    elif cfg.data.tokenizer == "ExponentialTimeTokenizer" or cfg.data.tokenizer == "NoLossTokenizer":
        dataset_name = "ExponentialTimeTokenDataset"
        tokenizer = ExponentialTimeTokenizer(**dataset_config["tokenizer_parameters"])
    elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
        dataset_name = "AwesomeTokensDataset"
        min_time_unit = dataset_config["tokenizer_parameters"]["min_time_unit"]
        n_velocity_bins = dataset_config["tokenizer_parameters"]["n_velocity_bins"]
        tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
        tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)

    dataset_split = st.text_input(label="split", value="validation+test")

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

    with st.expander("train config"):
        st.json(train_config)

    with st.expander("dataset config"):
        st.write(dataset_config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]
    source = json.loads(record["source"])

    with st.expander(label="source"):
        st.json(source)

    val_dataset = NextTokenDataset(dataset=dataset, tokenizer=tokenizer)
    eval_iters = cfg.eval_iters

    def get_batch():
        data = val_dataset
        ix = np.random.randint(0, len(data), size=(cfg.data.batch_size,))
        # numpy to int :(
        x = torch.stack([data[int(i)]["source_token_ids"] for i in ix])
        y = torch.stack([data[int(i)]["target_token_ids"] for i in ix])
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out["validation"] = losses.mean()
        model.train()
        return out

    st.write(estimate_loss())


if __name__ == "__main__":
    main()
