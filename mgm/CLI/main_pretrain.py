import torch
from torch.utils.data import random_split
import pandas as pd
from pickle import load
from mgm.src.MicroCorpus import MicroTokenizer
from mgm.src.MicroCorpus import MicroCorpus

import os
import warnings
from transformers import (
    BertForMaskedLM,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import EarlyStoppingCallback

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")  # For tensor cores

def pretrain(cfg, args):
    corpus = load(open(args.input, 'rb'))
    tokenizer = corpus.tokenizer
    # set model

    config = {
        "hidden_size": cfg.getint('Bert', 'hidden_size'),
        "num_hidden_layers": cfg.getint('Bert', 'num_hidden_layers'),
        "initializer_range": cfg.getfloat('Bert', 'initializer_range'),
        "layer_norm_eps": cfg.getfloat('Bert', 'layer_norm_eps'),
        "attention_probs_dropout_prob": cfg.getfloat('Bert', 'attention_probs_dropout_prob'),
        "hidden_dropout_prob": cfg.getfloat('Bert', 'hidden_dropout_prob'),
        "intermediate_size": cfg.getint('Bert', 'intermediate_size'),
        "hidden_act": cfg.get('Bert', 'hidden_act'),
        "max_position_embeddings": cfg.getint('Bert', 'max_position_embeddings'),
        "model_type": cfg.get('Bert', 'model_type'),
        "num_attention_heads": cfg.getint('Bert', 'num_attention_heads'),
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": tokenizer.vocab_size,
    }

    config = BertConfig(**config)
    
    training_args = {
    "learning_rate": cfg.getfloat('pretrain', 'learning_rate'),
    "do_train": True,
    "do_eval": True,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": "linear",
    "warmup_steps": cfg.getint('pretrain', 'warmup_steps'),
    "weight_decay": cfg.getfloat('pretrain', 'weight_decay'),
    "per_device_train_batch_size": cfg.getint('pretrain', 'per_device_train_batch_size'),
    "num_train_epochs": cfg.getint('pretrain', 'num_train_epochs'),
    "evaluation_strategy": "steps",
    "eval_steps": cfg.getint('pretrain', 'eval_steps'),
    "save_strategy": "steps",
    'save_steps': cfg.getint('pretrain', 'save_steps'),
    "logging_steps": cfg.getint('pretrain', 'logging_steps'),
    "output_dir": f'{args.log}/pretrain_checkpoints',
    "logging_dir": args.log,
    "load_best_model_at_end": True,
    }
    
    training_args = TrainingArguments(**training_args)
    
    print(f"Start training...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    # set a new model when mask rate changes
    model = BertForMaskedLM(config)
    model = model.train()
    
    split = args.val_split

    train_set, val_set = random_split(
        corpus, [int(len(corpus) * (1 - split)), len(corpus) - int(len(corpus) * (1 - split))]
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)

    # save logs
    logs = trainer.state.log_history
    logs = pd.DataFrame(logs)
    os.makedirs(args.log, exist_ok=True)
    logs.to_csv(os.path.join(args.log, "pretrain_log.csv"), index=False)