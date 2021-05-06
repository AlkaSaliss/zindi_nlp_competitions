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
from transformers import LongformerConfig, LongformerForMaskedLM, LongformerTokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
import time
import datetime
import json
import argparse


# fix randomness
seed_all()


def tokenize_and_train_longformer_lm(input_path, input_path_val, output_path, vocab_size=30_000, min_freq=2,
                                  max_len=256, block_size=64, mlm_probability=0.15,
                                  num_attention_heads=6, num_hidden_layers=3, epochs=5,
                                  batch_size=30, val_batch_size=60, eval_steps=50, attention_window=512, **kwargs):
    # instantiate tokenizer
    bpe_tokenizer = ByteLevelBPETokenizer()
    # train tokenizer
    _pretty_print("Training tokenizer")
    bpe_tokenizer.train([input_path, input_path_val], vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    # save tokenizer
    tok_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tok_path, exist_ok=True)
    bpe_tokenizer.save_model(tok_path)

    # load tokenizer with Longformer configuration
    bpe_tokenizer = LongformerTokenizerFast.from_pretrained(tok_path, max_len=max_len)

    # create data objects
    dataset_gen = LineByLineTextDataset(
        tokenizer=bpe_tokenizer,
        file_path=input_path,
        block_size=block_size
    )
    dataset_gen_val = LineByLineTextDataset(
        tokenizer=bpe_tokenizer,
        file_path=input_path_val,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bpe_tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # create model
    config = LongformerConfig(
        attention_window=attention_window,
        sep_token_id=bpe_tokenizer.get_vocab()["</s>"],
        pad_token_id=bpe_tokenizer.get_vocab()["<pad>"],
        bos_token_id=bpe_tokenizer.get_vocab()["<s>"], 
        eos_token_id=bpe_tokenizer.get_vocab()["</s>"],
        vocab_size=bpe_tokenizer.vocab_size,
        max_position_embeddings=max_len+10,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1
    )
    
    model = LongformerForMaskedLM(config=config)

    _pretty_print(f"Number of model parameters : {model.num_parameters():,}")

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
        eval_dataset=dataset_gen_val
    )

    # train model
    _pretty_print("Training starts")
    trainer.train()
    # save model
    trainer.save_model(model_path)

    return tok_path, model_path


if __name__ == "__main__":
    start = time.time()
    c_time = c_time = str(datetime.datetime.fromtimestamp(start)).split(".")[0]\
        .replace("-", "_")\
        .replace(" ", "_")\
        .replace(":", "_")
    parser = argparse.ArgumentParser(
        description="Train Longformer tokenizer and language model")
    parser.add_argument("config", help="training configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    flag = config.get("flag", "longformer_lm")
    output_path = config["output_path"]
    output_path = os.path.join(output_path, f"{flag}-{c_time}")
    os.makedirs(output_path, exist_ok=True)
    config["output_path"] = output_path

    tok_path, model_path = tokenize_and_train_longformer_lm(**config)
    config["model_path"] = model_path
    config["tokenizer_path"] = tok_path

    duration = time.time() - start
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    _pretty_print(f"Training took {h}h : {m}mn : {s}s ")

    config["duration"] = duration
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f)
