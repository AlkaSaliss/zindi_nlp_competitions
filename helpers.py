import os
import random
import numpy as np
import torch
from functools import reduce
import re


def _pretty_print(msg):
	print("+-+"*30)
	print("\t", msg)
	print("+-+"*30+"\n")

SEED = 123


def seed_all():
    # function to set seed
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def compose_proc_funcs(list_funcs):
    return reduce(lambda f1, f2: lambda text: f2(f1(text)), list_funcs, lambda text: text) 

NOISE_PATTERN = re.compile(r"(\ufeff|\n|\t)")
SPACE_PATTERN = re.compile(r"\s{2,}")
REP_CHAR_PATTERN = re.compile(r"(\S)(\1{2,})")
REP_WORD_PATTERN = re.compile(r"(?:^|\s)(\w+)\s((?:\1\s+)+)\1(\s|$|\W)")


def clean_whitespace(text):
    return SPACE_PATTERN.sub(" ", text).strip()


def clean_noise(text):
    return NOISE_PATTERN.sub(" ", text)


def clean_char_rep(text):
    def _remove_rep_char(_matches):
        pat, rep = _matches.groups()
        return f" {pat} <rep> {len(rep)+1} "
    return REP_CHAR_PATTERN.sub(_remove_rep_char, text)


def clean_word_rep(text):
    def _remove_rep_word(matches):
        pat, rep, end = matches.groups()
        return f' {pat} <rep> {len(rep.split())+2}{end}'
    return REP_WORD_PATTERN.sub(_remove_rep_word, text)


def to_lower(text): return text.lower()


DICT_TEXT_PREPROCESSORS = {
    "clean_whitespace": clean_whitespace,
    "clean_noise": clean_noise,
    "clean_char_rep": clean_char_rep,
    "clean_word_rep": clean_word_rep,
    "to_lower": to_lower
}
