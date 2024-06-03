import os
import  pandas as pd
from mgm.CLI.CLI_utils import find_pkg_resource
from mgm.src.MicroCorpus import (
    MicroCorpus,
    MicroTokenizer,
    SequenceClassificationDataset,
)
from mgm.src.utils import CustomUnpickler, gen_num_sent
import torch
from transformers import Trainer, GPT2LMHeadModel
from pickle import load, dump

def generate(cfg, args):
    with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
    extended_tokenizer = load(open(f"{args.model}/tokenizer.pkl", "rb"))
    model = GPT2LMHeadModel.from_pretrained(args.model)
    bad_words = set(extended_tokenizer.vocab.values()) - set(tokenizer.vocab.values())
    bad_words = [[word] for word in bad_words]
    
    if args.prompt is not None:
        prompt = pd.read_csv(args.prompt, sep='\t', header=None).values
        sent = [['<bos>', disease[0]] for disease in prompt]
    else:
        print("No prompt provided. Generating random sentences.")
        sent = [['<bos>']]
        
    start = [extended_tokenizer.encode(sent, return_tensors='pt') for sent in sent]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    gen_sent = gen_num_sent(start,
                            model,
                            num_sent=args.num_samples,
                            tokenizer=extended_tokenizer,
                            bad_words=bad_words) 
    
    dump(gen_sent, open(args.output, "wb"))
