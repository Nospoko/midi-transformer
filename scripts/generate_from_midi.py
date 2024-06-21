import argparse
from contextlib import nullcontext

import torch
import pandas as pd
import fortepyan as ff
from omegaconf import OmegaConf, DictConfig
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers import MidiTokenizer, OneTimeTokenizer, ExponentialTimeTokenizer

from gpt2.model import GPT, GPTConfig
from tokenized_midi_datasets import OneTimeTokenDataset, AwesomeTokensDataset, ExponentialTimeTokenDataset


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device) -> tuple[GPT, MidiTokenizer, DictConfig]:
    checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
    train_config = checkpoint["config"]
    cfg = OmegaConf.create(train_config)

    if "dataset_name" in train_config["data"].keys():
        config_name = cfg.data.dataset_name
        if cfg.data.tokenizer == "OneTimeTokenizer":
            dataset_config = OneTimeTokenDataset.builder_configs[config_name].builder_parameters
            tokenizer = OneTimeTokenizer(**dataset_config["tokenizer_parameters"])
        elif cfg.data.tokenizer == "ExponentialTimeTokenizer":
            dataset_config = ExponentialTimeTokenDataset.builder_configs[config_name].builder_parameters
            tokenizer = ExponentialTimeTokenizer(**dataset_config["tokenizer_parameters"])
        elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
            dataset_config = AwesomeTokensDataset.builder_configs[config_name].builder_parameters
            min_time_unit = dataset_config["tokenizer_parameters"]["min_time_unit"]
            n_velocity_bins = dataset_config["tokenizer_parameters"]["n_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)
    else:
        dataset_config = cfg.dataset
        if cfg.data.tokenizer == "OneTimeTokenizer":
            tokenizer = OneTimeTokenizer(**dataset_config["tokenizer_parameters"])
        elif cfg.data.tokenizer == "ExponentialTimeTokenizer":
            tokenizer = ExponentialTimeTokenizer(**dataset_config["tokenizer_parameters"])
        elif cfg.data.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = dataset_config["tokenizer_parameters"]["min_time_unit"]
            n_velocity_bins = dataset_config["tokenizer_parameters"]["n_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)

    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=dataset_config["sequence_length"],
        bias=cfg.model.bias,
        vocab_size=None,
        dropout=cfg.model.dropout,
    )
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    print("initializing model")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, cfg, dataset_config


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg, dataset_config = load_model_and_tokenizer(args.model_path, device)
    print("putting model on gpu")
    model.to(device)
    dataset_config = OmegaConf.create(dataset_config)
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

    input_piece = ff.MidiPiece.from_file(args.midi_path)
    input_notes = input_piece.df
    input_sequence = tokenizer.encode(input_notes)
    it = 0
    generated_noteframes = []
    total_sequences = len(input_sequence) // dataset_config.sequence_length
    offset = 0
    while it + dataset_config.sequence_length < len(input_sequence):
        sequence = input_sequence[it : it + dataset_config.sequence_length]
        original_notes = tokenizer.decode(sequence)

        sequence = torch.tensor([sequence], device=device)
        if it / (dataset_config.sequence_length) % 10 == 0:
            print(f"generating frame {it // dataset_config.sequence_length} / {total_sequences} ...")
        with torch.no_grad():
            with ctx:
                output = model.generate(
                    idx=sequence,
                    max_new_tokens=dataset_config["sequence_length"],
                    temperature=args.temperature,
                )

        output: torch.Tensor = output[0].cpu().numpy()

        out_notes = tokenizer.decode(output)
        out_notes = out_notes[len(original_notes) :]

        out_notes.end -= out_notes.start.min() - offset
        out_notes.start -= out_notes.start.min() - offset

        offset = out_notes.end.max()

        generated_noteframes.append(out_notes)
        it += dataset_config.sequence_length

    out_notes = pd.concat(generated_noteframes, ignore_index=True, axis=0).reset_index()
    out_piece = ff.MidiPiece(out_notes)

    output_midi_path = args.output_path
    out_piece.to_midi().write(output_midi_path)

    print(f"Generated MIDI saved to {output_midi_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI sequences from a pre-trained model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--midi_path",
        type=str,
        required=True,
        help="Path to the input MIDI file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated MIDI file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        help="Temperature parameter for model.generate",
        default=1.0,
    )

    args = parser.parse_args()
    main(args)
