import sys
import os
import pathlib
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
src_path = pathlib.Path.cwd().parent
sys.path.append(str(src_path))

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor
import torch.nn as nn
import torch.cuda.amp as amp
import torch
import argparse
from tqdm.auto import tqdm
import json
import datetime
import shutil
import time
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from transformers import RobertaTokenizerFast
from data_utils.text_data_loaders import map_label, get_roberta_dataloaders, get_roberta_inference
from helpers import _pretty_print, seed_all



seed_all()


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def create_supervised_trainer_with_pretraining(
        model,
        optimizer,
        loss_fn,
        device=None,
        non_blocking=False,
        prepare_batch=_prepare_batch,
        output_transform=lambda x, y, y_pred, loss: loss.item(),
        epochs_pretrain=None,
        mixed_precision=False):

    grad_scaler = amp.GradScaler() if mixed_precision else None

    def _update(engine, batch):
        # the model must have a module named base_model
        if epochs_pretrain is not None:
            if engine.state.epoch == 1:  # don't train base model for the first epochs
                for p in model.base_model.parameters():
                    p.requires_grad = False
            if engine.state.epoch == epochs_pretrain+1:  # set all params to trainable after pretraining
                for p in model.base_model.parameters():
                    p.requires_grad = True

        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        if mixed_precision:
            with amp.autocast():
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update)
    return trainer


def run_training(model, optimizer, scheduler, output_path,
                 train_loader, val_loader, epochs, patience, epochs_pretrain, mixed_precision):

    # trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crit = nn.CrossEntropyLoss()
    metrics = {"accuracy": Accuracy(), "loss": Loss(crit)}
    trainer = create_supervised_trainer_with_pretraining(
        model, optimizer, crit, device=device, epochs_pretrain=epochs_pretrain,
        mixed_precision=mixed_precision)
    train_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)

    # Out paths
    path_ckpt = os.path.join(output_path, "model_ckpt")
    log_dir = os.path.join(output_path, "log_dir")
    os.makedirs(log_dir, exist_ok=True)

    # tensorboard
    tb_logger = TensorboardLogger(log_dir=log_dir)
    tb_logger.attach(train_evaluator, log_handler=OutputHandler(tag="training", metric_names=[
        "accuracy", "loss"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag="validation", metric_names=[
        "accuracy", "loss"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

    # training progress
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)
        train_loss = train_evaluator.state.metrics["loss"]
        val_loss = val_evaluator.state.metrics["loss"]
        train_acc = train_evaluator.state.metrics["accuracy"]
        val_acc = val_evaluator.state.metrics["accuracy"]
        pbar.log_message(
            "Training Results - Epoch: {}  Loss: {:.6f}  Accuracy: {:.6f}".format(engine.state.epoch, train_loss, train_acc))
        pbar.log_message(
            "Validation Results - Epoch: {}  Loss: {:.6f}  Accuracy: {:.6f}".format(engine.state.epoch, val_loss, val_acc))

        pbar.n = pbar.last_print_n = 0

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)

    # def get_val_loss(engine):
    # 	return -engine.state.metrics['loss']
    def get_val_acc(engine):
        return engine.state.metrics['accuracy']

    # checkpoint and early stopping
    checkpointer = ModelCheckpoint(
        path_ckpt, "model", score_function=get_val_acc, score_name="accuracy", require_empty=False)
    early_stopper = EarlyStopping(patience, get_val_acc, trainer)

    to_save = {'optimizer': optimizer, 'model': model}
    if scheduler is not None:
        to_save["scheduler"] = scheduler
    val_evaluator.add_event_handler(Events.COMPLETED, checkpointer, to_save)
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopper)
    if scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # free resources
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda _: _empty_cache())
    train_evaluator.add_event_handler(
        Events.ITERATION_COMPLETED, lambda _: _empty_cache())
    val_evaluator.add_event_handler(
        Events.ITERATION_COMPLETED, lambda _: _empty_cache())

    trainer.run(train_loader, max_epochs=epochs)
    tb_logger.close()

    # Evaluation with best model
    model.load_state_dict(torch.load(
        glob.glob(os.path.join(path_ckpt, "*.pth"))[0])["model"])
    train_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)

    train_evaluator.run(train_loader)
    val_evaluator.run(val_loader)

    _pretty_print("Evaluating best model")
    pbar.log_message(
        "Best model on training set - Loss: {:.6f}  Accuracy: {:.6f}"
        .format(train_evaluator.state.metrics["loss"], train_evaluator.state.metrics["accuracy"]))
    pbar.log_message(
        "Best model on validation set - Loss: {:.6f}  Accuracy: {:.6f}"
        .format(val_evaluator.state.metrics["loss"], val_evaluator.state.metrics["accuracy"]))

    return model, train_evaluator.state.metrics, val_evaluator.state.metrics


def run_inference(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    preds = []
    with torch.no_grad():
        for x_ids, x_mask in tqdm(test_loader):
            pred = model((x_ids.to(device), x_mask.to(device)))
            preds.append(pred.cpu())
    preds = torch.cat(preds)
    _, preds = torch.topk(preds, 1, dim=1)
    preds = [map_label(item, True) for item in preds.numpy().flatten()]

    return preds


if __name__ == "__main__":
    start = time.time()
    c_time = str(datetime.datetime.fromtimestamp(start)).split(".")[0]\
        .replace("-", "_")\
        .replace(" ", "_")\
        .replace(":", "_")

    parser = argparse.ArgumentParser(
        description="Trains transformer based model for classification")
    parser.add_argument("config", type=str,
                        help="Path to the json configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        train_config = json.load(f)

    # prepare dataset
    input_path = train_config["input_path"]
    test_path = train_config["test_path"]
    output_path = train_config["output_path"]
    model_path = train_config["model_path"]
    epochs = train_config["epochs"]
    epochs_pretrain = train_config.get("epochs_pretrain")
    mixed_precision = train_config.get("mixed_precision", False)
    patience = train_config["patience"]
    flag = train_config.get("flag", "xp")
    output_path = os.path.join(output_path, f"{flag}-{c_time}")
    os.makedirs(output_path, exist_ok=True)
    test_size = train_config.get("test_size", 0.2)
    list_procs = train_config["list_procs"]
    text_col = train_config["text_col"]
    label_col = train_config["label_col"]
    tokenizer_path = train_config["tokenizer_path"]
    max_length = train_config["max_length"]
    train_batch_size = train_config["train_batch_size"]
    val_batch_size = train_config["val_batch_size"]
    val_batch_size = train_config["val_batch_size"]
    num_workers = train_config["num_workers"]
    pin_memory = train_config.get("pin_memory", False)

    _pretty_print("Loading data")
    df = pd.read_csv(input_path)
    df_test = pd.read_csv(test_path)
    df_train, df_val = train_test_split(
        df, test_size=test_size, random_state=123, stratify=df[label_col])

    tokenizer = RobertaTokenizerFast.from_pretrained(
        tokenizer_path, max_len=max_length)
    train_loader, val_loader = get_roberta_dataloaders(df_train, df_val, list_procs,
                                                       text_col, label_col, tokenizer, max_length,
                                                       train_batch_size, val_batch_size, num_workers, pin_memory)

    # load model
    dict_model = {}
    with open(model_path) as f:
        exec(f.read(), dict_model)
    model = dict_model["model"]
    optimizer = dict_model["optimizer"]
    scheduler = dict_model.get("scheduler")
    shutil.copy(model_path, os.path.join(output_path, "model_script.py"))
    train_config["model_script"] = os.path.join(output_path, "model_script.py")

    _pretty_print("Training start")
    model, train_metrics, val_metrics = run_training(model, optimizer, scheduler,
                                                     output_path, train_loader,
                                                     val_loader, epochs, patience,
                                                     epochs_pretrain, mixed_precision)
    end1 = time.time()
    duration1 = end1 - start
    m, s = divmod(duration1, 60)
    h, m = divmod(m, 60)
    _pretty_print(f"Training took {h}h : {m}mn : {s}s ")

    _pretty_print("Inference start")
    test_loader = get_roberta_inference(df_test, list_procs, text_col, tokenizer, max_length,
                                        val_batch_size, num_workers, pin_memory)

    preds = run_inference(model, test_loader)

    df_sub = pd.DataFrame({
        "ID": df_test["ID"],
        "label": preds
    })
    sub_path = os.path.join(output_path, f"sub_{flag}-{c_time}.csv")
    df_sub.to_csv(sub_path, index=False)

    duration2 = time.time() - end1
    m, s = divmod(duration2, 60)
    h, m = divmod(m, 60)
    _pretty_print(f"Inference took {h}h : {m}mn : {s}s ")

    # save config
    train_config["train_metrics"] = train_metrics
    train_config["val_metrics"] = val_metrics
    train_config["train_time"] = duration1
    train_config["inference_time"] = duration2
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(train_config, f, indent=2)
