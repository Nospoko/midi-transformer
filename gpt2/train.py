"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import math
import time
from contextlib import nullcontext

import hydra
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from midi_tokenizers_generation.tokenizer_generator import generate_tokenizer

from gpt2.model import GPT, GPTConfig
from data.next_token_dataset import NextTokenDataset

load_dotenv()
tokenizer_name_to_dataset_map: dict[str, str] = {
    "ExponentialTimeTokenizer": "ExponentialTimeTokenDataset",
    "OneTimeTokenizer": "OneTimeTokenDataset",
    "AwesomeMidiTokenizer": "AwesomeTokensDataset",
}


@hydra.main(config_path="configs", config_name="gpt2_noloss_pretraining")
def main(cfg: DictConfig):
    out_dir = to_absolute_path(cfg.out_dir)
    if cfg.task == "pretraining":
        out_dir = os.path.join(
            out_dir,
            "pretraining",
        )
    # Get the right data for the tokenizer specified in config
    dataset_name = tokenizer_name_to_dataset_map[cfg.data.tokenizer]
    dataset_config = cfg.dataset
    tokenizer_parameters = dataset_config.tokenizer_parameters

    # Hydra changes paths - we have to change them back
    # Load the suitable dataset
    dataset_path = to_absolute_path(f"./tokenized_midi_datasets/{dataset_name}")
    dataset = load_dataset(
        dataset_path,
        num_proc=8,
        trust_remote_code=True,
        **dataset_config,
    )
    total_tokens = dataset_config.sequence_length * dataset["train"].num_rows
    print(f"tokens in a training dataset: {total_tokens}")

    # Keep config as a dict as well for logging at wandb and for checkpoints
    config = OmegaConf.to_container(cfg)
    if cfg.data.tokenizer == "AwesomeMidiTokenizer":
        tokenizer_path = to_absolute_path("pretrained/awesome_tokenizers/awesome-tokenizer-pretrained.json")
        tokenizer = AwesomeMidiTokenizer.from_file(tokenizer_path)
    else:
        tokenizer = generate_tokenizer(name=cfg.data.tokenizer, parameters=tokenizer_parameters)

    train_dataset = NextTokenDataset(dataset=dataset["train"], tokenizer=tokenizer)
    val_dataset = NextTokenDataset(dataset=dataset["validation"], tokenizer=tokenizer)

    device = cfg.system.device

    # Various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.ddp.backend)

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

        seed_offset = ddp_rank  # each process gets a different seed

        # World_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.data.gradient_accumulation_steps % ddp_world_size == 0
        cfg.data.gradient_accumulation_steps //= ddp_world_size

    else:
        # If not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_batch = cfg.data.batch_size * dataset_config.sequence_length
    tokens_per_iter = cfg.data.gradient_accumulation_steps * ddp_world_size * tokens_per_batch
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def get_batch(split):
        if split == "train":
            data = train_dataset
        else:
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

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_vocab_size = tokenizer.vocab_size

    print(f"found vocab_size = {meta_vocab_size} (inside {tokenizer.name})")

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

    if cfg.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        model_args["vocab_size"] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif cfg.init_from == "resume":
        print(f"Resuming training from {out_dir}")

        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
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
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    elif cfg.init_from.startswith("midi-gpt2"):
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, f"pretrained/{cfg.init_from}")
        checkpoint = torch.load(ckpt_path, map_location=device)
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

    # crop down the model block size if desired, using model surgery
    if dataset_config.sequence_length < model.config.block_size:
        model.crop_block_size(dataset_config.sequence_length)
        model_args["block_size"] = dataset_config.sequence_length  # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        device_type=device_type,
    )

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model is never used ...
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    milion_params = model.get_num_params() / 1e6
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < cfg.lr.warmup_iters:
            return cfg.optimizer.learning_rate * it / cfg.lr.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr.lr_decay_iters:
            return cfg.lr.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.lr.warmup_iters) / (cfg.lr.lr_decay_iters - cfg.lr.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.lr.min_lr + coeff * (cfg.optimizer.learning_rate - cfg.lr.min_lr)

    run_name = f"midi-gpt2-{milion_params:.0f}M-" + cfg.logging.wandb_run_name_suffix
    # logging
    if cfg.logging.wandb_log and master_process:
        wandb.init(project=cfg.logging.wandb_project, name=run_name, config=config)
        # define our custom x axis metric
        wandb.define_metric("total_tokens")
        # define which metrics will be plotted against it
        wandb.define_metric("train_batch/loss", step_metric="total_tokens")
        wandb.define_metric("val_batch/loss", step_metric="total_tokens")
        wandb.define_metric("train/loss", step_metric="total_tokens")

    total_tokens = 0
    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    iter_num = 1
    while True:
        current_tokens = X.numel()
        # total_tokens += current_tokens
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.lr.decay_lr else cfg.optimizer.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train_batch/loss": losses["train"],
                        "val_batch/loss": losses["val"],
                        "total_tokens": total_tokens,
                    }
                )
            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, run_name + ".pt"))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        t00 = time.time()
        n_iter_tokens = 0
        for micro_step in range(cfg.data.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == cfg.data.gradient_accumulation_steps - 1
            with ctx:
                n_iter_tokens += X.numel()
                logits, loss = model(X, Y)
                # scale the loss to account for gradient accumulation
                loss = loss / cfg.data.gradient_accumulation_steps

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        total_tokens += n_iter_tokens

        # clip the gradient
        if cfg.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        t_forward_backward = time.time() - t00

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.logging.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.data.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            # I'm not sure about this
            tokens_per_second = current_tokens / dt

            # Here's my version
            tps = n_iter_tokens / t_forward_backward
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "total_tokens": total_tokens,
                    "tps": tps,
                    "tokens_per_second": tokens_per_second,
                }
            )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                + f"time {dt:.2f}s, mfu {running_mfu*100:.2f}%, "
                + f"tokens_per_second {tokens_per_second:.2f} "
                + f"tps {tps:.2f}"
            )
        iter_num += 1
        local_iter_num += 1

        if iter_num == cfg.optimizer.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
