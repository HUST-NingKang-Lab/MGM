from mgm.CLI.CLI_utils import get_CLI_parser, get_CFG_reader
from mgm.src.MicroCorpus import MicroCorpus, MicroTokenizer
from mgm.src.utils import seed_everything
import sys
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Starting MGM...")
    parser = get_CLI_parser()
    args = parser.parse_args()
    cfg = get_CFG_reader(args.config)
    
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)
    else:
        print("No seed provided, the program will generate a random seed.")
        
    if args.mode == 'construct':
        from mgm.CLI.main_construct import construct
        construct(cfg, args)
        sys.exit(0)
    elif args.mode == 'pretrain':
        from mgm.CLI.main_pretrain import pretrain
        pretrain(cfg, args)
        sys.exit(0)
    elif args.mode == 'train':
        from mgm.CLI.main_train import train
        train(cfg, args)
        sys.exit(0)
    elif args.mode == 'finetune':
        from mgm.CLI.main_finetune import finetune
        finetune(cfg, args)
        sys.exit(0)
    elif args.mode == 'predict':
        from mgm.CLI.main_predict import predict
        predict(cfg, args)
        sys.exit(0)
    elif args.mode == 'generate':
        from mgm.CLI.main_generate import generate
        generate(cfg, args)
        sys.exit(0)
    elif args.mode == 'reconstruct':
        from mgm.CLI.main_reconstruct import reconstruct
        reconstruct(cfg, args)
        sys.exit(0)
    else:
        raise RuntimeError('Please specify correct work mode, see `--help`.')

#main()
