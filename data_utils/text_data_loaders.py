import sys
import os
import pathlib
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
src_path = pathlib.Path.cwd().parent
sys.path.append(str(src_path))
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from helpers import compose_proc_funcs, seed_worker, DICT_TEXT_PREPROCESSORS


def map_label(label, is_reverse):
    dict_mapping = {
        0: 0,
        1: 1,
        -1: 2
    }
    dict_mapping_reverse = {v: k for k, v in dict_mapping.items()}
    if is_reverse:
        return dict_mapping_reverse[label]
    return dict_mapping[label]


class TextDataGenerator(Dataset):

    def __init__(self, df, text_col, label_col=None, list_procs=None):

        self.list_texts = list(df[text_col])
        self.label_col = label_col
        if label_col is not None:
            self.list_labels = list(df[label_col])

        list_proc_funcs = []
        if list_procs is not None:
            for func_name in list_procs:
                assert func_name in DICT_TEXT_PREPROCESSORS.keys(),\
                    f"processing function {func_name} not found in  DICT_TEXT_PREPROCESSORS"
                list_proc_funcs.append(DICT_TEXT_PREPROCESSORS[func_name])
        self.proc_func = compose_proc_funcs(list_proc_funcs)

    def __len__(self):
        return len(self.list_texts)

    def __getitem__(self, idx):
        text = self.proc_func(self.list_texts[idx])
        if self.label_col is not None:
            label = self.list_labels[idx]
            label = map_label(label, False)
            return text, label
        return text


def _collate_fn_roberta(batch, tokenizer, max_length, with_label=True):
    if with_label:
        text, label = list(zip(*batch))
        text = tokenizer.batch_encode_plus(
            list(text), max_length=max_length, truncation=True, padding=True)
        input_ids = torch.LongTensor(text["input_ids"])
        attention_mask = torch.LongTensor(text["attention_mask"])
        label = torch.LongTensor(label)
        return (input_ids, attention_mask), label
    text = list(batch)
    text = tokenizer.batch_encode_plus(
        text, max_length=max_length, truncation=True, padding=True)
    input_ids = torch.LongTensor(text["input_ids"])
    attention_mask = torch.LongTensor(text["attention_mask"])
    return input_ids, attention_mask


def get_roberta_dataloaders(df_train, df_val, list_procs,
                            text_col, label_col, tokenizer, max_length,
                            train_batch_size, val_batch_size, num_workers, pin_memory):
    train_dataset = TextDataGenerator(
        df_train, text_col, label_col, list_procs)
    val_dataset = TextDataGenerator(df_val, text_col, label_col, list_procs)

    collate_fn = partial(_collate_fn_roberta,
                         tokenizer=tokenizer, max_length=max_length, with_label=True)

    train_loader = DataLoader(train_dataset, train_batch_size,
                              shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, val_batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, worker_init_fn=seed_worker)
    return train_loader, val_loader


def get_roberta_inference(df, list_procs, text_col, tokenizer, max_length,
                          batch_size, num_workers, pin_memory):
    inf_dataset = TextDataGenerator(df, text_col, None, list_procs)
    collate_fn = partial(_collate_fn_roberta,
                         tokenizer=tokenizer, max_length=max_length, with_label=False)
    inf_loader = DataLoader(inf_dataset, batch_size,
                            shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, worker_init_fn=seed_worker)
    return inf_loader
