placeholder_tokens = [f"<SENTINEL_{idx}>" for idx in range(98)]
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
    "<LOW_FROM_MEDIAN>" "<HIGH_FROM_MEDIAN>",
] + placeholder_tokens


prediction_task_to_token_pair = {
    "bass_prediction": ("<BASS>", "<NO_BASS>"),
    "reverse_bass_prediction": ("<NO_BASS>", "<BASS>"),
    "tenor_prediction": ("<TENOR>", "<NO_TENOR>"),
    "alto_prediction": ("<ALTO>", "<NO_ALTO>"),
    "soprano_prediction": ("<SOPRANO>", "<NO_SOPRANO>"),
    "treble_prediction": ("<TREBLE>", "<NO_TREBLE>"),
    "ppp_prediction": ("<PPP>", "<NO_PPP>"),
    "pp_prediction": ("<PP>", "<NO_PP>"),
    "p_prediction": ("<P>", "<NO_P>"),
    "mp_prediction": ("<MP>", "<NO_MP>"),
    "mf_prediction": ("<MF>", "<NO_MF>"),
    "f_prediction": ("<F>", "<NO_F>"),
    "high_median_prediction": ("<LOW_FROM_MEDIAN>", "<HIGH_FROM_MEDIAN>"),
}

voice_task_to_range = {
    "bass_prediction": (21, 48),
    "reverse_bass_prediction": (49, 108),
    "tenor_prediction": (43, 81),
    "alto_prediction": (53, 84),
    "soprano_prediction": (60, 96),
    "treble_prediction": (60, 108),
}

dynamic_task_to_range = {
    "ppp_prediction": (0, 30),
    "pp_prediction": (30, 50),
    "p_prediction": (50, 70),
    "mp_prediction": (70, 90),
    "mf_prediction": (90, 110),
    "f_prediction": (110, 127),
}


def get_source_task_token(prediction_task: str):
    return prediction_task_to_token_pair[prediction_task][1]


def get_target_task_token(prediction_task: str):
    return prediction_task_to_token_pair[prediction_task][0]


def get_voice_task_range(task: str):
    return voice_task_to_range[task]


def get_velocity_range(dynamic_instruction: str):
    return dynamic_task_to_range[dynamic_instruction]
