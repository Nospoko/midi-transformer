import time

import pandas as pd
from datasets import load_dataset
from midi_tokenizers import ExponentialTimeTokenizer


def main():
    """Test the speed of encoding step in tokenizer"""
    tokenizer = ExponentialTimeTokenizer()
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="test+train+validation")
    pieces = [pd.DataFrame(record["notes"]) for record in dataset]
    tokens_total = 0
    print("starting")
    t0 = time.time()
    for notes in pieces:
        tokens = tokenizer.encode(notes=notes)
        tokens_total += len(tokens)
    t_total = time.time() - t0

    print(f"tokens per second: {(tokens_total / t_total):.3f}")


if __name__ == "__main__":
    main()
