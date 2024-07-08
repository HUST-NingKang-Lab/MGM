import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mgm.src.evaluator import Evaluator
import os
import pickle
from mgm.src.MicroCorpus import MicroTokenizer, MicroCorpus
from tqdm import tqdm

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def eval_and_save(y_score, y_true, label_names, save_dir, activation="softmax"):
    if activation == "sigmoid":
        y_score = nn.Sigmoid()(torch.tensor(y_score)).numpy()
    elif activation == "softmax":
        y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    elif activation == "none":
        y_score = torch.tensor(y_score).numpy()
    # y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    evaluator = Evaluator(y_score, y_true, label_names=label_names, num_thresholds=100)
    metrics, avg_metrics = evaluator.eval()

    # save
    for label, metric in metrics.items():
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        metric.to_csv(os.path.join(save_dir, f"{label}.csv"))
    avg_metrics.to_csv(os.path.join(save_dir, "avg.csv"))
    return avg_metrics

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MicroTokenizer':
            return MicroTokenizer
        return super().find_class(module, name)

def generate(sent, model, tokenizer, do_sample=True, bad_words_ids=None, num_return_sequences=100):
    sent = sent.to(model.device)
    gen_sent = model.generate(sent, 
                                max_length=512, 
                                do_sample=do_sample,
                                bad_words_ids=bad_words_ids,
                                num_return_sequences=num_return_sequences,
                                forced_eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                low_memory=True if num_return_sequences > 1 else False)
    return gen_sent.cpu().detach()

def gen_num_sent(start, model, num_sent, tokenizer, bad_words=None):
    gen_sent = [generate(sent, model, tokenizer, bad_words_ids=bad_words, num_return_sequences=num_sent) for sent in start]
    gen_sent = [torch.cat([sent, torch.ones(num_sent, 512 - sent.shape[1], dtype=torch.long) * tokenizer.pad_token_id], dim=1) for sent in gen_sent]
    gen_sent = torch.cat(gen_sent, dim=0)
    return gen_sent

def loss_bc(p_i,q_i):
    return torch.sum(torch.abs(p_i-q_i))/torch.sum(torch.abs(p_i+q_i))

def get_Z(corpus, position_encodings, vocab_size, label=True):
    
    if label:
        corpus = torch.cat((corpus[:, 0:1], corpus[:, 2:],
                            torch.zeros(corpus.shape[0], 1, dtype=torch.long)), 
                            dim=1)
    
    Z = torch.zeros((len(corpus), vocab_size))
    idx = corpus
    for i in tqdm(range(len(corpus))):
        running_idx = idx[i][idx[i] > 3]
        Z[i, running_idx] = 1 + position_encodings[:len(running_idx)]

    return Z[:, 4:]




