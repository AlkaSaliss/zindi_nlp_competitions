import sys
import pathlib
src_path = pathlib.Path.cwd().parent
sys.path.append(str(src_path))
from helpers import DICT_TEXT_PREPROCESSORS, compose_proc_funcs
from tqdm.auto import tqdm
import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create training data")
    parser.add_argument("--files", help="list of file paths", nargs="+",
                        default=[
                            "/home/alka/Documents/zindi_challenges/zindi_text_classif/inputs/tunisian_sent_an_arabizi/Train.csv",
                            "/home/alka/Documents/zindi_challenges/zindi_text_classif/inputs/tunisian_sent_an_arabizi/Test.csv"
                        ])
    parser.add_argument(
        "--text_col", help="name of text column in files", default="text")
    parser.add_argument(
        "--test_split", help="percentage of validation data, 0<test_split<1", default=0.15, type=float)
    parser.add_argument(
        "--procs", help="list of processor names", nargs="+", default=["clean_noise", "clean_whitespace"])
    parser.add_argument("--flag", help="name of output folder", required=True)
    parser.add_argument(
        "--output_path", help="root folder where to save data", required=True)
    args = parser.parse_args()

    # load data
    list_texts = sum([list(pd.read_csv(p, usecols=[args.text_col])[args.text_col])
                      for p in args.files], [])
    # processing functions
    list_proc_funcs = []
    if args.procs != []:
        for func_name in args.procs:
            assert func_name in DICT_TEXT_PREPROCESSORS.keys(),\
                f"processing function {func_name} not found in  DICT_TEXT_PREPROCESSORS"
            list_proc_funcs.append(DICT_TEXT_PREPROCESSORS[func_name])
    proc_func = compose_proc_funcs(list_proc_funcs)
    # preprcess text
    list_texts = list(map(proc_func, tqdm(list_texts)))

    # split train-val
    list_texts_train, list_texts_val = train_test_split(
        list_texts, test_size=args.test_split, random_state=123)

    output_path = os.path.join(args.output_path, args.flag)
    os.makedirs(output_path, exist_ok=True)
    config = {
        "files": args.files,
        "text_col": args.text_col,
        "test_split": args.test_split,
        "procs": args.procs,
        "flags": args.flag,
        "output_path": args.output_path
    }

    # save train and val texts
    with open(os.path.join(output_path, "raw_text_train.txt"), "w") as f:
        f.writelines(f"{item}\n" for item in list_texts_train)
    with open(os.path.join(output_path, "raw_text_val.txt"), "w") as f:
        f.writelines(f"{item}\n" for item in list_texts_val)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
