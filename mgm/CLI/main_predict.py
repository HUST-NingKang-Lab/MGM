import os
import  pandas as pd
from mgm.CLI.CLI_utils import find_pkg_resource
from mgm.src.MicroCorpus import (
    MicroCorpus,
    MicroTokenizer,
    SequenceClassificationDataset,
)
from mgm.src.utils import eval_and_save

import torch
from transformers import Trainer, GPT2ForSequenceClassification
from pickle import load, dump
        
def predict(cfg, args):
    corpus = load(open(args.input, "rb"))
    tokenizer = corpus.tokenizer
    
    if args.evaluate:
        if args.labels is None:
            raise ValueError("Please provide labels for evaluation.")
        else:
            labels = pd.read_csv(args.labels, index_col=0)
            
            if set(corpus.data.index) != set(labels.index):
                # warning
                print(
                    "Warning: the sample IDs in the abundance table and the metadata table are not the same."
                    "The samples in the metadata table but not in the abundance table will be removed."
                    "This may happened because some samples were removed for all zero counts during the preprocessing of the abundance table."
                )
            labels = labels.loc[corpus.data.index]
            le = load(open(f'{args.model}/label_encoder.pkl', 'rb'))
            labels = le.transform(labels.values.reshape(-1, 1)).toarray()
            labels = torch.tensor(labels.argmax(axis=1))
            
            dataset = SequenceClassificationDataset(
                corpus[:]["input_ids"], corpus[:]["attention_mask"], labels
            )
    else:
        print("Only predict the labels, no evaluation. Please pay attention to the threshold for manual evaluation.")
        le = load(open(f'{args.model}/label_encoder.pkl', 'rb'))
        labels = [0]*len(corpus)
        dataset = SequenceClassificationDataset(
            corpus[:]["input_ids"], corpus[:]["attention_mask"], labels
        )

    model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=len(le.categories_[0]))
    model.eval()
    trainer = Trainer(model=model)
    
    predictions = trainer.predict(dataset)
    
    # save predictions
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    y_score = predictions.predictions
    pd.DataFrame(y_score, index=corpus.data.index, columns=le.categories_[0]).to_csv(f'{args.output}/y_score.csv')
    
    if args.evaluate:
        y_true = predictions.label_ids
        eval_and_save(y_score, y_true, le.categories_[0], f'{args.output}/evaluation')
