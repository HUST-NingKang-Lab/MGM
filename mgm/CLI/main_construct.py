from mgm.src.MicroCorpus import MicroCorpus, MicroTokenizer
from mgm.CLI.CLI_utils import find_pkg_resource
from mgm.src.utils import CustomUnpickler
from pickle import load, dump, Unpickler
import sys


def construct(cfg, args):
    # tokenizer = load(open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb"))
    with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
        
    if not args.no_normalize:
        print("Your data will be normalized with the phylogeny mean and std. If you wish to use your own normalization, please use --no-normalize.")
    corpus = MicroCorpus(
        data_path=args.input,
        tokenizer=tokenizer,
        key=args.key,
        max_len=cfg.getint("construct", "max_len"),
        preprocess=not args.no_normalize,
    )
    dump(corpus, open(args.output, "wb"))
