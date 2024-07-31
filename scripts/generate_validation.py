import os
import argparse
from contextlib import nullcontext

import torch

import gpt2.utils as gpt2_utils


def main(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(f=model_path, map_location=device)

    cfg = gpt2_utils.load_cfg(checkpoint=checkpoint)
    tokenizer = gpt2_utils.load_tokenizer(cfg)

    model = gpt2_utils.initialize_model(
        cfg,
        checkpoint=checkpoint,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    validation_examples = gpt2_utils.prepare_validation_examples_for_task(cfg)

    run_name = os.path.splitext(os.path.basename(model_path))[0]
    gpt2_utils.run_generation_step(
        model=model,
        checkpoint=checkpoint,
        run_name=run_name,
        validation_examples=validation_examples,
        tokenizer=tokenizer,
        device=device,
        ctx=ctx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on validation examples.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("device", type=str, help="Device to perform calculations on")
    args = parser.parse_args()

    main(args.model_path, args.device)
