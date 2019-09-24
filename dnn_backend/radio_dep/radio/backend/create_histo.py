#!/usr/bin/python

import os
import sys
import glob
from os.path import join
from os.path import dirname
import argparse
import pandas as pd

from radio import dataset as ds
from radio import CTImagesMaskedBatch
from radio.pipelines.backend import accumulate_histo
from radio.utils import DEFAULT_CONFIG as CONFIG


def check_args(args):
    """ Check passed commandline arguments. """
    if not (args.batch_size > 0 and isinstance(args.batch_size, int)):
        sys.exit("Batch size must be positive integer")

    if not len(glob.glob(args.dataset)) > 1:
        sys.exit("Argument 'dataset' must be regexp for dataset files")

    if args.fmt not in ('dicom', 'raw', 'blosc'):
        sys.exit("Incorrect format: must be 'blosc', 'raw' or 'dicom'")

    if not os.path.exists(args.annotation):
        sys.exit("Incorrect path to file with annotation")
    else:
        try:
            annotation = pd.read_csv(args.annotation)
        except Exception:
            sys.exit("Unable to read file with annotation")

        columns = ['coordZ', 'coordY', 'coordX', 'diameter_mm', 'seriesuid']
        if any(c not in annotation.columns for c in columns):
            sys.exit("Annotation must be provided in csv format with " +
                     "table containing following columns: {}".format(columns))

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "This script allows to create dataset" +
        " of crops that will be used for" +
        " artificial neural networks learning.")
    )

    parser.add_argument('-d', '--dataset', type=str,
                        help='Paths to dicom or mhd scans.')

    parser.add_argument('-a', '--annotation', type=str,
                        help='Path to file with annotation.')

    parser.add_argument('-o', '--output', type=str, default=join(os.getcwd(), 'histo.pkl'),
                        help='Path to file where histogram will be saved.')

    parser.add_argument('--batch_size', type=int,
                        default=CONFIG.accumulate_histo.batch_size,
                        help='Size of batch containing scans.' +
                        ' Default is {}.'.format(CONFIG.accumulate_histo.batch_size))

    parser.add_argument('--fmt', type=str, default='dicom',
                        help='Format of input data. Can be "dicom" or "raw".' +
                        ' Default is "dicom".')

    args = parser.parse_args()
    # Checking passed arguments
    check_args(args)

    config_update = {
        "accumulate_histo": {
            'batch_size': args.batch_size
        }
    }
    scans_index = ds.FilesIndex(path=args.dataset,
                                dirs=(args.fmt != 'raw'),
                                no_ext=(args.fmt == 'raw'))

    scans_dataset = ds.Dataset(scans_index,
                               batch_class=CTImagesMaskedBatch)
    accumulate_histo(scans_dataset, args.fmt,
                     pd.read_csv(args.annotation),
                     output_path=args.output,
                     config=config_update)
