#!/usr/bin/python

import os
import sys
import glob
import argparse

import pandas as pd

import torch

from radio import CTImagesMaskedBatch
from radio import dataset as ds
from radio.pipelines.backend import evaluate_model
from radio.utils import DEFAULT_CONFIG as CONFIG


RADIO_EXPERIMENTS_PATH = CONFIG.GLOBAL['RADIO_EXPERIMENTS_PATH']


def check_args(args):

    if not os.path.exists(os.path.join(RADIO_EXPERIMENTS_PATH, args.experiment, 'model.tar')):
        sys.exit("No experiment found with name {}".format(args.experiment))

    if args.task not in ('classification', 'segmentation'):
        sys.exit("Task must be 'classification' or 'segmentation'. See -t option.")

    if args.fmt not in ('dicom', 'raw', 'blosc'):
        sys.exit("Incorrect format: must be 'blosc', 'raw' or 'dicom'")

    if not (args.scans_batch_size > 0 and isinstance(args.scans_batch_size, int)):
        sys.exit(
            "Scans batch size must be positive integer. See --scans_batch_size option.")

    if not (args.crops_batch_size > 0 and isinstance(args.crops_batch_size, int)):
        sys.exit("Crops batch size must be positive integer." +
                 " See --crop_batch_size option.")

    if args.device < 0 or args.device >= torch.cuda.device_count():
        sys.exit("Incorrect device id. Must be positive" +
                 " integer [0...{}]".format(torch.cuda.device_count() - 1) +
                 " See --device option.")

    if args.mask_mode not in ('ellipsoid', 'cuboid'):
        sys.exit("Incorrect mask creation mode: must be 'ellipsoid' or 'cuboid'")

    if not args.dataset:
        sys.exit("Dataset path must be provided")
    elif not all(os.path.exists(p) for p in glob.glob(args.dataset)):
        sys.exit("Incorrect path to dataset: {}".format(args.dataset))
    elif not len(glob.glob(args.dataset)) > 1:
        sys.exit("Argument 'dataset' must be regexp for dataset files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "This script allows to validate pretrained Deep Neural Networks.")
    )

    parser.add_argument('-d', '--dataset', type=str,
                        help="Paths to dicom or mhd scans.")

    parser.add_argument('-a', '--annotation', type=str,
                        help="Path to file with annotation.")

    parser.add_argument('-e', '--experiment', type=str,
                        help="Name of experiment.")

    parser.add_argument('-t', '--task', type=str, default='segmentation',
                        help=("Type of task. Can be 'segmentation'" +
                              " or 'classification'."))

    parser.add_argument('--scans_batch_size', type=int,
                        default=CONFIG.evaluation.scans_batch_size,
                        help=('Size of batch containing scans.' +
                              ' Default value is {}.'.format(CONFIG
                                                             .evaluation
                                                             .scans_batch_size)))

    parser.add_argument('--crops_batch_size', type=int,
                        default=CONFIG.evaluation.predict_on_scan.batch_size,
                        help=('Size of batch containing crops' +
                              ' that will be used for prediction.' +
                              ' Default value is {}.'.format(CONFIG
                                                             .evaluation
                                                             .predict_on_scan
                                                             .batch_size)))

    parser.add_argument('--fmt', type=str, default='dicom',
                        help=('Format of input scans data.' +
                              ' Can be "dicom" or "raw".'))

    parser.add_argument('--mask_mode', type=str,
                        default=CONFIG.preprocessing.mask_mode,
                        help='Shape of mask. Can be "ellipsoid" or "cuboid".' +
                        ' Default is "{}".'.format(CONFIG
                                                   .preprocessing
                                                   .mask_mode))

    parser.add_argument('--padding_mode', type=str,
                        default=CONFIG.preprocessing.unify_spacing.padding,
                        help='Mode of padding. Can be "reflect",' +
                        ' "edge", "zeros" or "constant". ' +
                        ' Default is "{}".'.format(CONFIG
                                                   .preprocessing
                                                   .mask_mode))

    parser.add_argument('--device', type=int, default=0,
                        help=('GPU device number that will be used' +
                              ' for model training. Default is 0.'))

    args = parser.parse_args()

    # Checking passed arguments
    check_args(args)

    config_update = {
        'preprocessing': {
            'mask_mode': args.mask_mode,
            'unify_spacing': {
                'padding': (args.padding_mode
                            if args.padding_mode != "zeros"
                            else "constant")
            }
        },
        'evaluation': {
            'scans_batch_size': args.scans_batch_size,
            'predict_on_scan': {
                'batch_size': args.crops_batch_size
            }
        }
    }

    scans_index = ds.FilesIndex(path=args.dataset,
                                dirs=(args.fmt != 'raw'),
                                no_ext=(args.fmt == 'raw'))

    evaluate_model(
        fmt=args.fmt,
        dataset=ds.Dataset(index=scans_index,
                           batch_class=CTImagesMaskedBatch),
        nodules=pd.read_csv(args.annotation), config=config_update,
        model=torch.load(os.path.join(RADIO_EXPERIMENTS_PATH, args.experiment,
                                      'model.tar')).eval().train(True),
        device=args.device, task=args.task,
        save_path=os.path.join(RADIO_EXPERIMENTS_PATH,
                               args.experiment)
    )
