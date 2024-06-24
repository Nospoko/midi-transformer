"""
    TBH, I think keeping ranges of voices and velocities as a dataset parameter and experimenting with
    tiny variations is tragically boring. I would much prefer keeping voice definintions the same and experimenting
    with various combinations of extraction types instead.
    That is why in this file I will keep all the voice definitions, extraction type definitions etc.
    Let me know if that makes sense.
"""
placeholder_tokens = [f"<SENTINEL_{idx}>" for idx in range(100)]
special_tokens = [
    "<CLS>",
    "<EOS>",
    "<PAD>",
    "<RANDOM>",
    "<PPP>",
    "<PP>",
    "<P>",
    "<MP>",
    "<MF>",
    "<F>",
    "<BASS>",
    "<TENOR>",
    "<ALTO>",
    "<SOPRANO>",
    "<TREBLE>",
    "<NO_RANDOM>",
    "<NO_PPP>",
    "<NO_PP>",
    "<NO_P>",
    "<NO_MP>",
    "<NO_MF>",
    "<NO_F>",
    "<NO_BASS>",
    "<NO_TENOR>",
    "<NO_ALTO>",
    "<NO_SOPRANO>",
    "<NO_TREBLE>",
] + placeholder_tokens

pitch_types = ["bass", "tenor", "alto", "soprano", "treble"]
extraction_type_to_token_pair = {
    "bass": ("<BASS>", "<NO_BASS>"),
    "tenor": ("<TENOR>", "<NO_TENOR>"),
    "alto": ("<ALTO>", "<NO_ALTO>"),
    "soprano": ("<SOPRANO>", "<NO_SOPRANO>"),
    "treble": ("<TREBLE>", "<NO_TREBLE>"),
    "ppp": ("<PPP>", "<NO_PPP>"),
    "pp": ("<PP>", "<NO_PP>"),
    "p": ("<P>", "<NO_P>"),
    "mp": ("<MP>", "<NO_MP>"),
    "mf": ("<MF>", "<NO_MF>"),
    "f": ("<F>", "<NO_F>"),
}

extraction_type_to_range = {
    "bass": (21, 48),
    "tenor": (43, 81),
    "alto": (53, 84),
    "soprano": (60, 96),
    "treble": (60, 108),
    "ppp": (0, 30),
    "pp": (30, 50),
    "p": (50, 70),
    "mp": (70, 90),
    "mf": (90, 110),
    "f": (110, 127),
}
