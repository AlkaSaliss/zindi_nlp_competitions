import sys
import os
import pathlib

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
src_path = pathlib.Path.cwd().parent
sys.path.append(str(src_path))
from helpers import _pretty_print, seed_all
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import (
    FunnelConfig,
    FunnelForMaskedLM,
    FunnelTokenizerFast,
    FunnelTokenizer,
    PreTrainedTokenizerFast,
)
from tokenizers import ByteLevelBPETokenizer
import time
import datetime
import json
import argparse


# fix randomness
seed_all()


def tokenize_and_train_funnel_lm(
    input_path,
    input_path_val,
    output_path,
    vocab_size=30_000,
    min_freq=2,
    max_len=256,
    block_size=64,
    block_sizes=[4, 4, 4],
    mlm_probability=0.15,
    num_attention_heads=6,
    epochs=5,
    batch_size=30,
    val_batch_size=60,
    eval_steps=50,
    **kwargs,
):
    # instantiate tokenizer
    bpe_tokenizer = ByteLevelBPETokenizer()
    # train tokenizer
    _pretty_print("Training tokenizer")
    bpe_tokenizer.train(
        [input_path, input_path_val],
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            "<sep>",
            "<cls>"
        ],
    )
    # save tokenizer
    tok_path = os.path.join(output_path, "tokenizer")
    tok_path_file = os.path.join(tok_path, "vocab.json")
    os.makedirs(tok_path, exist_ok=True)
    # bpe_tokenizer.save_model(tok_path)
    bpe_tokenizer.save(tok_path_file, True)

    # load tokenizer with Roberta configuration
    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path_file,
        max_length=max_len,
        lowercase=True,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
    )
    # bpe_tokenizer = FunnelTokenizerFast(
    #     vocab_file=tok_path,
    #     max_length=max_len,
    #     lowercase=True,
    #     sep_token="<sep>",
    #     pad_token="<pad>",
    #     cls_token="<cls>",
    #     mask_token="<mask>",
    #     bos_token="<s>",
    #     eos_token="</s>",
    # )

    # create data objects
    dataset_gen = LineByLineTextDataset(
        tokenizer=bpe_tokenizer, file_path=input_path, block_size=block_size
    )
    dataset_gen_val = LineByLineTextDataset(
        tokenizer=bpe_tokenizer, file_path=input_path_val, block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bpe_tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # create model
    config = FunnelConfig(
        vocab_size=bpe_tokenizer.vocab_size,
        max_position_embeddings=max_len + 10,
        n_head=num_attention_heads,
        block_sizes=block_sizes,
        type_vocab_size=1,
    )
    model = FunnelForMaskedLM(config=config)

    _pretty_print(f"Number of model parameters : {model.num_parameters()}")

    model_path = os.path.join(output_path, "lm")
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=val_batch_size,
        evaluation_strategy="steps",
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        greater_is_better=False,
        fp16=True,
        # dataloader_num_workers=4,
        # max_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_gen,
        eval_dataset=dataset_gen_val,
    )

    # train model
    _pretty_print("Training starts")
    trainer.train()
    # save model
    trainer.save_model(model_path)

    return tok_path, model_path


if __name__ == "__main__":
    start = time.time()
    c_time = c_time = (
        str(datetime.datetime.fromtimestamp(start))
        .split(".")[0]
        .replace("-", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )
    parser = argparse.ArgumentParser(
        description="Train funnel-transformer tokenizer and language model"
    )
    parser.add_argument("config", help="training configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    flag = config.get("flag", "funnel_lm")
    output_path = config["output_path"]
    output_path = os.path.join(output_path, f"{flag}-{c_time}")
    os.makedirs(output_path, exist_ok=True)
    config["output_path"] = output_path

    tok_path, model_path = tokenize_and_train_funnel_lm(**config)
    config["model_path"] = model_path
    config["tokenizer_path"] = tok_path

    duration = time.time() - start
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    _pretty_print(f"Training took {h}h : {m}mn : {s}s ")

    config["duration"] = duration
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f)
