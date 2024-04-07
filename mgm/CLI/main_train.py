# run under the root directory of the project
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from mgm.src.MicroCorpus import (
    MicroCorpus,
    MicroTokenizer,
    SequenceClassificationDataset,
)

from pickle import load, dump
from sklearn.preprocessing import OneHotEncoder
from transformers import (
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    GPT2Config,
)
from transformers.trainer_callback import EarlyStoppingCallback


def train(cfg, args):
    corpus = load(open(args.input, "rb"))
    tokenizer = corpus.tokenizer
    labels = pd.read_csv(args.labels, index_col=0)
    if set(corpus.data.index) != set(labels.index):
        # warning
        print(
            "Warning: the sample IDs in the abundance table and the metadata table are not the same."
            "The samples in the metadata table but not in the abundance table will be removed."
            "This may happened because some samples were removed for all zero counts during the preprocessing of the abundance table."
        )
    labels = labels.loc[corpus.data.index]

    # label encoding
    le = OneHotEncoder()
    labels = le.fit_transform(labels.values.reshape(-1, 1)).toarray()
    labels = torch.tensor(labels.argmax(axis=1))

    # packing into dataset
    dataset = SequenceClassificationDataset(
        corpus[:]["input_ids"], corpus[:]["attention_mask"], labels
    )

    # set model config
    config = {
        'model_type':cfg.get('GPT2', 'model_type'),
        'vocab_size':tokenizer.vocab_size,
        'n_positions':cfg.getint('GPT2', 'n_positions'),
        'n_embd': cfg.getint('GPT2', 'n_embd'),
        'n_layer': cfg.getint('GPT2', 'n_layer'),
        'n_head': cfg.getint('GPT2', 'n_head'),
        'bos_token_id':tokenizer.bos_token_id,
        'eos_token_id':tokenizer.eos_token_id,
        'pad_token_id':tokenizer.pad_token_id,
    }
    

    config = GPT2Config(**config)
    config.num_labels = le.categories_[0].shape[0]
    model = GPT2ForSequenceClassification(config)

    # set training args
    training_args = {
        "learning_rate": cfg.getfloat("train", "learning_rate"),
        "do_train": True,
        "do_eval": True,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": "linear",
        "warmup_steps": cfg.getint("train", "warmup_steps"),
        "weight_decay": cfg.getfloat("train", "weight_decay"),
        "per_device_train_batch_size": cfg.getint("train", "per_device_train_batch_size"),
        "num_train_epochs": cfg.getint("train", "num_train_epochs"),
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "logging_steps": cfg.getint("train", "logging_steps"),
        "output_dir": f"{args.log}/train_checkpoints",
        "logging_dir": args.log,
        "load_best_model_at_end": True,
    }

    training_args = TrainingArguments(**training_args)

    print(f"Start training...")
    model = model.train()

    split = args.val_split

    train_set, val_set = random_split(
        dataset, [int(len(corpus) * (1 - split)), int(len(corpus) * split)]
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        callbacks=callbacks,
    )

    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    dump(le, open(os.path.join(args.output, "label_encoder.pkl"), "wb"))
    

    # save logs
    logs = trainer.state.log_history
    logs = pd.DataFrame(logs)
    os.makedirs(args.log, exist_ok=True)
    logs.to_csv(os.path.join(args.log, "train_log.csv"), index=False)
