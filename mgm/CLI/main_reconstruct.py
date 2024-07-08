
import pandas as pd
import numpy as np
import os
import torch
from pickle import load
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mgm.CLI.CLI_utils import find_pkg_resource
from mgm.src.MicroCorpus import MicroCorpus
from mgm.src.utils import CustomUnpickler, get_Z
from mgm.src.Reconstructor import reconstructorNet, PositionEmbedding

#input: abundance_file.csv; model.ckpt; sentence.pkl
#output: reconstrcuted_abundance; model.ckpt



def reconstruct(cfg, args):
    
    retrain = True
    if args.abu is None :
        if args.reconModel is None:
            raise ValueError("Please provide abundace for train or load trained model.")
        else:
            model = reconstructorNet.load_from_checkpoint(args.reconModel, N=9665, lr=0.0002)
            retrain = False
    else:
        if args.reconModel is None:
            with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
                unpickler = CustomUnpickler(f)
                tokenizer = unpickler.load()
            
            corpus = MicroCorpus(
                data_path=args.abu,
                tokenizer=tokenizer,
                key='genus',
                max_len=cfg.getint("construct", "max_len"),
                preprocess=not args.no_normalize)
            vocab_size = corpus.tokenizer.vocab_size 
        else:
            raise ValueError("Don't need to provide abundance and model both.")
    
    
    if retrain:
        #Get abundance and corpus data and split for train
        abundance = corpus.data
        corpus = corpus[:]['input_ids']
        position_embedding = PositionEmbedding()   
        position_encodings = position_embedding(corpus[0]).reshape(-1)
        
        P_train = abundance
        P_train = torch.tensor(P_train.values)
        Z_train = corpus.data
        Z_train = get_Z(Z_train, position_encodings, vocab_size, label=args.withLabel)

        del corpus

        # Initial model
        training_args = {
            "learning_rate": cfg.getfloat("reconstruct", "learning_rate"),
            "max_epochs": cfg.getint("reconstruct", "num_train_epochs"),
            "batch_size":cfg.getint("reconstruct", "per_device_train_batch_size")
        }

        mb = training_args["batch_size"]
        lr = training_args["learning_rate"]
        epochs = training_args["max_epochs"]

        train_set = TensorDataset(Z_train, P_train)
        train_set, val_set = random_split(train_set, [0.8, 0.2])
        train_loader = DataLoader(train_set, batch_size=mb, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=mb)

        N = Z_train.shape[1]
        model = reconstructorNet(N, lr)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

        # Training and save model
        grad_clip_val = 1.0 
        trainer = Trainer(max_epochs=epochs, callbacks=callbacks, gradient_clip_val=grad_clip_val)
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(os.path.join(args.output,'reconstructor_model.ckpt'))

    
    
    # Generate reconstructed abundance
    if args.input is None:
        raise ValueError('Please provide ranked corpus for reconstruct model.')
    
    ordered_corpus = load(open(args.input, 'rb'))
    with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
                unpickler = CustomUnpickler(f)
                tokenizer = unpickler.load()
    vocab_size = tokenizer.vocab_size
    
    if args.withLabel :
        if args.generator is None:
            raise ValueError('Please provide generator for load tokenizer.')
        extented_tokenizer = load(open(f'{args.generator}/tokenizer.pkl', 'rb'))
        labels = [extented_tokenizer.decode([i]) for i in ordered_corpus[:, 1]]
        labelsframe = pd.DataFrame(data=labels, columns=['label'])
        labelsframe.to_csv(os.path.join(args.output,'label.csv'), index=False)
        
    
    position_embedding = PositionEmbedding()   
    position_encodings = position_embedding(ordered_corpus[0]).reshape(-1)
    
    pred_args = {
            "learning_rate": cfg.getfloat("reconstruct", "learning_rate"),
            "max_epochs": cfg.getint("reconstruct", "num_train_epochs"),
            "batch_size":cfg.getint("reconstruct", "per_device_train_batch_size")
        }

    mb = pred_args["batch_size"]
    
    gen = get_Z(ordered_corpus, position_encodings, vocab_size, args.withLabel)
    gen_set = TensorDataset(gen)
    gen_loader = DataLoader(gen_set, batch_size=mb)
    
    pred = []
    for batch in tqdm(gen_loader):
        p = batch
        p_pred = model.predict(p[0])
        pred.append(p_pred)
    
    pred = torch.cat(pred, 0)
    pred = pred / pred.mean(dim=1, keepdim=True)
    pred = pred.detach().cpu().numpy()
    pd.DataFrame(pred).to_csv(os.path.join(args.output,'reconstructed_abundance.csv'), index=False)
    
    


