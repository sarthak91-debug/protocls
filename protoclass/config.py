import os
import argparse
from pathlib import Path
from easydict import EasyDict as edict


"""
Arguments for creating the dataset & parameters

"""


def model_parse_args():


    parser=argparse.ArgumentParser(
        prog='ModelParser',
        description='Model and Data Arguments'
    )
    parser.add_argument('--save_path',default='../data/llm_train.csv',type=str)
        
    parser.add_argument('--raw_path',default='../data/train.json',type=str)
    
    
    parser.add_argument('--test_path',default='../data/test.csv',type=str)

    parser.add_argument('--val_path',default='../data/val.csv',type=str)

        
    parser.add_argument('--model',default='sentence-transformers/all-MiniLM-L12-v2',type=str)
    parser.add_argument('--batch_size',default=32,type=int)
    
    parser.add_argument('--init',default='random',type=str)
    
    parser.add_argument('--n_classes',default=5,type=int)
    parser.add_argument('--n_protos',default=5,type=int)
    parser.add_argument('--pos_protos',default=5,type=int)
    parser.add_argument('--neg_protos',default=1,type=int)

    args=parser.parse_args()
    
    return args



def log_parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - args: Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog='BenchmarksParser',
        description="Benchmark Classification Metrics")
    parser.add_argument(
        '--log_file',
        type=lambda p: Path(p).absolute(),
        default=Path('classification_metrics.log').absolute(),
        help='Path to the log file. Default is "classification_metrics.log" in the current directory.'
    )
    parser.add_argument(
        '--true_labels',
        type=Path,
        required=True,
        help='Path to the file containing true labels.'
    )
    parser.add_argument(
        '--pred_labels',
        type=Path,
        required=True,
        help='Path to the file containing predicted labels.'
    )
    parser.add_argument(
        '--probabilities',
        type=Path,
        help='Path to the file containing predicted probabilities. Required for Log Loss and ROC AUC.'
    )
    
    args=parser.parse_args()
    
    return args


