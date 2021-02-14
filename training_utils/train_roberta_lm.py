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
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
import time
import json
import argparse


# fix randomness
seed_all()


def tokenize_and_train_roberta_lm(input_path, output_path, vocab_size=30_000, min_freq=2,
                                  max_len=512, block_size=64, mlm_probability=0.15,
                                  num_attention_heads=12, num_hidden_layers=6, epochs=5,
                                  batch_size=30, **kwargs):
    # instantiate tokenizer
    bpe_tokenizer = ByteLevelBPETokenizer()
    # train tokenizer
    _pretty_print("Training tokenizer")
    bpe_tokenizer.train(input_path, vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<rep>"
    ])
    # save tokenizer
    tok_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tok_path, exist_ok=True)
    bpe_tokenizer.save_model(tok_path)

    # load tokenizer with Roberta configuration
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained(
        tok_path, max_len=max_len)

    # create data objects
    dataset_gen = LineByLineTextDataset(
        tokenizer=bpe_tokenizer,
        file_path=input_path,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bpe_tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # create model
    config = RobertaConfig(
        vocab_size=bpe_tokenizer.vocab_size,
        max_position_embeddings=max_len+10,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1
    )
    model = RobertaForMaskedLM(config=config)

    _pretty_print(f"Number of model parameters : {model.num_parameters()}")

    model_path = os.path.join(output_path, "lm")
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # save_steps=50,
        logging_steps=50,
        save_total_limit=1,
        # fp16=True,
        # dataloader_num_workers=4,
        # max_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_gen,
    )

    # train model
    _pretty_print("Training starts")
    trainer.train()
    # save model
    trainer.save_model(model_path)

    return tok_path, model_path


if __name__ == "__main__":
    start = time.time()
    c_time = str(int(start))
    parser = argparse.ArgumentParser(
        description="Train Roberta tokenizer and language model")
    parser.add_argument("config", help="training configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    flag = config.get("flag", "roberta_lm")
    output_path = config["output_path"]
    output_path = os.path.join(output_path, f"{flag}-{c_time}")
    os.makedirs(output_path, exist_ok=True)
    config["output_path"] = output_path

    tok_path, model_path = tokenize_and_train_roberta_lm(**config)
    config["model_path"] = model_path
    config["tokenizer_path"] = tok_path

    duration = time.time() - start
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    _pretty_print(f"Training took {h}h : {m}mn : {s}s ")

    config["duration"] = duration
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f)