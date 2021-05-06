import sys
import pathlib
src_path = pathlib.Path.cwd().parent
sys.path.append(str(src_path))
import os
import json
import argparse
import torch
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast
from data_utils.text_data_loaders import (
    map_label,
    get_roberta_dataloaders,
    get_roberta_inference,
)
from helpers import _pretty_print, seed_all



def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def run_inference(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    preds = []
    with torch.no_grad():
        for x_ids, x_mask in tqdm(test_loader):
            pred = model((x_ids.to(device), x_mask.to(device)))
            preds.append(pred.cpu())
            del pred
            _empty_cache()
    preds = torch.cat(preds)
    # _, preds = torch.topk(preds, 1, dim=1)
    # preds = [map_label(item, True) for item in preds.numpy().flatten()]
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict probas from kfold")
    parser.add_argument("xp", help="experiment folder")
    parser.add_argument("--bs", help="batch_size", default=128, type=int)
    args = parser.parse_args()
    
    with open(os.path.join(args.xp, "config.json")) as f:
        config = json.load(f)

    df_test = pd.read_csv(config["test_path"]).reset_index(drop=True)
    df_sub = pd.DataFrame({"ID": df_test["ID"]})
    list_all_preds = []

    tokenizer = RobertaTokenizerFast.from_pretrained(config["tokenizer_path"], max_len=config["max_length"])
    test_loader = get_roberta_inference(
        df_test,
        config["list_procs"],
        config["text_col"],
        tokenizer,
        config["max_length"],
        args.bs,
        config["num_workers"],
        config["pin_memory"]
    )

    list_ckpts = glob.glob(os.path.join(args.xp, "fold*/model_ckpt/*pt*"))

    load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for ckpt in tqdm(list_ckpts):
        dict_model = {}
        with open(os.path.join(args.xp, "model_script.py")) as f:
            exec(f.read(), dict_model)
        model = dict_model["model"]
        model.load_state_dict(torch.load(ckpt, map_location=load_device)["model"])
        preds = run_inference(model, test_loader)
        list_all_preds.append(preds)
        del model; _empty_cache()

    final_preds = torch.nn.functional.softmax(torch.stack(list_all_preds).mean(dim=0), dim=1).numpy()
    with open(config["dict_label_path"]) as f:
        dict_label = json.load(f)
    reverse_dict_label = {v: k for k, v in dict_label.items()}
    list_cols = [reverse_dict_label[i] for i in range(final_preds.shape[1])]
    df_sub[list_cols] = final_preds

    output_path = os.path.join(args.xp, pathlib.Path(args.xp).name+".csv")
    df_sub.to_csv(output_path, index=False)


    




