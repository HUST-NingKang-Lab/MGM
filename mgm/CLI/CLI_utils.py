import argparse
import os
from configparser import ConfigParser
import pkg_resources
from typing import Optional

def get_CFG_reader(cfg_path: Optional[str]=None):
    cfg = ConfigParser()
    if cfg_path is not None:
        cfg.read(cfg_path)
        return cfg
    print('No config file provided, use default config file.')
    assert pkg_resources.resource_exists('mgm', 'resources/config.ini')
    cfg.read(pkg_resources.resource_filename('mgm', 'resources/config.ini'))
    return cfg


def find_pkg_resource(path):
    if pkg_resources.resource_exists('mgm', path):
        return pkg_resources.resource_filename('mgm', path)
    else:
        raise FileNotFoundError('Resource {} not found, please check'.format(path))

def get_CLI_parser():
    modes = ['construct', 'map','pretrain', 'train', 'finetune', 'predict']
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description=('MGM (Microbiao General Model) is a large-scaled pretrained language model for interpretable microbiome data analysis.\n'
                     'The program is designed to help you to fine-tune and evaluate MGM'
                     'to other  microbiome data analysis tasks.\n'
                     'Feel free to contact us if you have any question.\n'
                     'For more information, see Github. Thank you for using MGM!'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', type=str, default='predict', choices=modes,
                        help='The work mode for expert program.')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='The input file, see input format for each work mode.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='The output file, see output format for each work mode.')
    parser.add_argument('-k', '--key', type=str, default='genus',
                        help='The hdf5 key for input file.')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='The config.ini file.')
    parser.add_argument('-l', '--labels', type=str, default=None,
                        help='The path to csv file (storing labels for the input data).')
    parser.add_argument('-m', '--model', type=str, default=pkg_resources.resource_filename('mgm', 'resources/general_model'),
                        help='The path to MGM model. Could be self-superivsed or supervised model. Default is the pretrained MGM model on MGnify data.')
    parser.add_argument('-s', '--val-split', type=float, default=0.1,
                       help='The fraction of validation samples.')
    parser.add_argument('-H', '--log', type=str, default='MGM_log',
                       help='The path to store training history of MGM model.')

    # ------------------------------------------------------------------------------------------------------------------
    construct = parser.add_argument_group(
        title='construct', description='Convert input abundance data to countmatrix at Genus level and '
                                     'Normalize the countmatrix using phylogeny.\n'
                                     'Then construct a microbiome corpus using the countmatrix.\n'
                                     'Each sample is represented by a sentence from high rank genus to low rank genus.\n'
                                     'Input: the input data, Output: a pkl file containing the microbiome corpus.')
    construct.add_argument('--no-normalize', action='store_true', default=False,
                           help='Do not normalize the countmatrix if you have already normalized it by your own approach.')
    # ------------------------------------------------------------------------------------------------------------------
    pretrain = parser.add_argument_group(
        title = 'pretrain', description='Pretrain the MGM model using the microbiome corpus followed BERT style.\n'
                                        'Input: the microbiome corpus, Output: pretrained MGM model.')
    # ------------------------------------------------------------------------------------------------------------------

    train = parser.add_argument_group(
        title='train', description='Train supervised MGM model without mask pretrained weights.'
                                   'A microbiome corpus constructed by `construct` mode and properly labeled data '
                                   'must be provided. Label file should be in csv format, with the first column as sample names, second column as labels.\n'
                                   'Input: corpus in pkl file, label file in csv format, output: supervised MGM model')
    # ------------------------------------------------------------------------------------------------------------------
    finetune = parser.add_argument_group(
        title='finetune', description='Finetune MGM model to fit in a new ontology, A microbiome corpus '
                                      'and properly labeled data must be provided.\n'
                                      'use `-model` option to indicate a customized MGM model.\n'
                                      'Input: corpus in pkl file, label file in csv format, output: finetuned MGM model')

    # ------------------------------------------------------------------------------------------------------------------
    predict = parser.add_argument_group(
        title='predict', description='Predict the label of input data using the expert model.\n'
                                      'Input: corpus in pkl file, model file, output: prediction results')
    predict.add_argument('-E', '--evaluate', action='store_true', default=False,
                        help='Whether to evaluate the prediction results.')
        
    return parser