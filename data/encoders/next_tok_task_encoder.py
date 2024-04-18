from midi_tokenizers.midi_tokenizer import MidiTokenizer

from data.encoders.encoder import MidiEncoder


class NextTokTaskEncoder(MidiEncoder):
    """
    Encoder for next token prediction task.
    """

    def __init__(self, tokenizer: MidiTokenizer):
        self.tokenizer = tokenizer

    def encode(self, record: dict):
        tokens = record["notes"]
        encoding = [self.tokenizer.token_to_id[token] for token in tokens]

        # the inputs to the transformer will be the offset sequence
        src = encoding[:-1]
        tgt = encoding[1:]

        return src, tgt
