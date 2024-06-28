placeholder_tokens = [f"<SENTINEL_{idx}>" for idx in range(100)]
special_tokens = [
    "<PAD>",
    "<CLS>",
    "<EOS>",
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

voice_to_range = {
    "bass": (21, 48),
    "tenor": (43, 81),
    "alto": (53, 84),
    "soprano": (60, 96),
    "treble": (60, 108),
}

dynamic_to_range = {
    "ppp": (0, 30),
    "pp": (30, 50),
    "p": (50, 70),
    "mp": (70, 90),
    "mf": (90, 110),
    "f": (110, 127),
}


def get_source_extraction_token(extraction_type: str):
    return extraction_type_to_token_pair[extraction_type][1]


def get_target_extraction_token(extraction_type: str):
    return extraction_type_to_token_pair[extraction_type][0]


def get_voice_range(voice: str):
    return voice_to_range[voice]


def get_velocity_range(dynamic_instruction: str):
    return dynamic_to_range[dynamic_instruction]
