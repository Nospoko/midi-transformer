import os
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import fortepyan as ff
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from gpt2.model import GPT, GPTConfig
from tokenized_midi_datasets import OneTimeTokenDataset, AwesomeTokensDataset, ExponentialTimeTokenDataset


def list_files_in_directory(directory):
    directory_path = Path(directory)
    file_paths = [str(file) for file in directory_path.glob("**/*") if file.is_file()]
    return file_paths


def main():
    device = "cuda:0"
    run_name = "midi-gpt2-333M-pretraining-2024-04-30-14-59"

    checkpoint_path = f"checkpoints/pretrained/{run_name}.pt"
    if not os.path.exists(f"tmp/{run_name}"):
        os.mkdir(f"tmp/{run_name}")
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    checkpoint = torch.load(f=checkpoint_path, map_location=device)

    train_config = checkpoint["config"]
    cfg = OmegaConf.create(train_config)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.float16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    base_dataset_path = "roszcz/maestro-sustain-v2"
    dataset_split = "test+validation"

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

    # Generate parameters
    temperature = 1.0
    max_new_tokens = dataset_config.sequence_length
    if os.path.exists(f"tmp/{run_name}/file_descriptors.json"):
        with open(f"tmp/{run_name}/file_descriptors.json", "r+") as file:
            file_descriptors = json.load(file)
    else:
        file_descriptors = {}
    try:
        piece_counter = {}
        file_paths = os.listdir(f"tmp/{run_name}")
        for record in tqdm(dataset, total=len(dataset)):
            source = json.loads(record["source"])

            composer = source["composer"]
            title = source["title"]
            piece_name = (title + composer).replace(" ", "_").casefold()
            if piece_name in piece_counter.keys():
                piece_counter[piece_name] += 1
            else:
                piece_counter |= {piece_name: 0}

            notes = tokenizer.decode(record["note_token_ids"])
            source["original end"] = notes.end.max()

            idx = piece_counter[piece_name]
            if idx >= 2:
                continue

            filename = f"full__variations_on_{piece_name}_{idx}.mid".replace('"', "")
            filename = filename.replace("/", "_")
            full_midi_path = f"tmp/{run_name}/{filename}"
            file_descriptors |= {filename: source}

            if filename in file_paths:
                continue

            print(f"generating sample from {piece_name}")

            input_sequence = torch.tensor([record["note_token_ids"]], device=device)

            with torch.no_grad():
                with ctx:
                    output = model.generate(
                        idx=input_sequence,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

            output: torch.Tensor = output[0].cpu().numpy()
            # we want to decode the whole output so that the pressed notes can be unpressed by the tokenizer
            try:
                out_notes = tokenizer.decode(output)
            except IndexError:
                print(f"Problem: {source}")
                continue

            out_piece = ff.MidiPiece(out_notes, source=source)

            out_file = out_piece.to_midi()

            out_file.write(full_midi_path)

        file_descriptors_path = f"tmp/{run_name}/file_descriptors.json"
    finally:
        with open(file_descriptors_path, "+w") as file:
            json.dump(file_descriptors, file)


if __name__ == "__main__":
    main()
