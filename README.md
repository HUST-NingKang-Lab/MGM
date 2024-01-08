# MGM

MGM (Microbial General Model) is a large-scaled pretrained language model for interpretable microbiome data analysis. The program is designed to help you to fine-tune and evaluate MGM to other microbiome data analysis tasks.

## Installation

To install the MGM package, you can use setup.py:

```bash
python setup.py install
```

## Usage

You can use MGM in the command line interface (CLI) with different modes. The general usage is:

```bash
mgm <mode> [options]
```

The available modes are:

- `construct`: Convert input abundance data to countmatrix at Genus level and normalize the countmatrix using phylogeny. Then construct a microbiome corpus using the countmatrix. Each sample is represented by a sentence from high rank genus to low rank genus. Input: the input data in hdf5, csv or tsv format. Rows represent features and columns represent samples. Output: a pkl file containing the microbiome corpus.

e.g.:

```bash
mgm construct -i infant_data/abundance.csv -o infant_corpus.pkl
```
if you provide a hdf5 file, you need to specify the key of the input file using `-k` option. The default key is `genus`.

- `pretrain`: Pretrain the MGM model using the microbiome corpus followed BERT style. Input: corpus constructed by `construct` mode, Output: pretrained MGM model.

e.g.:

```bash
mgm pretrain -i infant_corpus.pkl -o infant_model
```

- `train`: Train supervised MGM model without mask pretrained weights. A microbiome corpus and properly labeled data must be provided. Label file should be in csv format, with the first column as sample names, second column as labels. Input: corpus constructed by `construct` mode. label file in csv format with the first column as sample names, second column as labels. Output: supervised MGM model.

e.g.:

```bash
mgm train -i infant_corpus.pkl -l infant_data/meta_withbirth.csv -o infant_model_clf
```

- `finetune`: Finetune MGM model to fit in a new ontology, A microbiome corpus and properly labeled data must be provided. Use `-model` option to indicate a customized MGM model. If not provided, the pretrained MGM model on MGnify data will be used. Input: corpus constructed by `construct` mode. label file in csv format with the first column as sample names, second column as labels. Pretrained model directory (Optional): finetuned MGM model.

e.g.:

```bash
mgm finetune -i infant_corpus.pkl -l infant_data/meta_withbirth.csv -m infant_model -o infant_model_clf_finetune
```

- `predict`: Predict the label of input data using the expert model. If the label file is provided with `-E`,the prediction results will be compared with the ground truth across different metrics. We recommend to use this mode to evaluate the performance of MGM model. If you want to perform your own evaluation, please pay attention to the thresholds.Input: corpus constructed by `construct` mode. label file (Optional) in csv format with the first column as sample names, second column as labels. Supervised MGM model. Output: prediction results in csv format.
  
e.g.:

```bash
mgm predict -E -i infant_corpus.pkl -l infant_data/meta_withbirth.csv -m infant_model_clf -o infant_prediction.csv
```

For more detailed usage, please refer to the help message of each mode:

```bash
mgm <mode> --help
```

## Maintainers
| Name | Email | Organization |
| ---- | ----- | ------------ |
|Haohong Zhang|[haohongzh@gmail.com](mailto:haohongzh@gmail.com)|PhD student, School of Life Science and Technology, Huazhong University of Science & Technology|
|Kang Ning  | [ningkang@hust.edu.cn](mailto:ningkang@hust.edu.cn)       | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |

