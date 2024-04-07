import torch
from torch.utils.data import random_split
import pandas as pd
from pickle import load
from mgm.src.MicroCorpus import MicroTokenizer
from mgm.src.MicroCorpus import MicroCorpus

import os
import warnings
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
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
    model = GPT2LMHeadModel(config)
    model = model.train()
    
    split = args.val_split

    train_set, val_set = random_split(
        corpus, [1-split, split]
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