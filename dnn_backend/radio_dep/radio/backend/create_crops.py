#!/usr/bin/python

import os
import sys
import argparse
import glob
import pandas as pd

from radio import dataset as ds
from radio import CTImagesMaskedBatch
from radio.utils import load_histo
from radio.utils import DEFAULT_CONFIG as CONFIG
from radio.pipelines.backend import create_crops


RADIO_CROPS_PATH = CONFIG.GLOBAL['RADIO_CROPS_PATH']


def check_args(args):
    """ Check passed commandline arguments. """
    if not (args.batch_size > 0 and isinstance(args.batch_size, int)):
        sys.exit("Batch size must be positive integer")

    if not (args.rate > 0 and isinstance(args.rate, (int, float))):
        sys.exit("Rate must be postive integer or float")

    if not args.dataset:
        sys.exit("Dataset path must be provided")
    elif not all(os.path.exists(p) for p in glob.glob(args.dataset)):
        sys.exit("Incorrect path to dataset: {}".format(args.dataset))
    elif not len(glob.glob(args.dataset)) > 1:
        sys.exit("Argument 'dataset' must be regexp for dataset files")

    if args.fmt not in ('dicom', 'raw', 'blosc'):
        sys.exit("Incorrect format: must be 'blosc', 'raw' or 'dicom'")

    if args.mask_mode not in ('ellipsoid', 'cuboid'):
        sys.exit("Incorrect mask creation mode: must be 'ellipsoid' or 'cuboid'")

    if args.padding_mode not in ('zeros', 'reflect', 'edge', 'constant'):
        sys.exit("Argument 'padding_mode' must be 'zeros'," +
                 " 'constant', 'reflect' or 'edge'")

    if not args.annotation:
        sys.exit("Path to csv file with annotation must be provided")
    elif not os.path.exists(args.annotation):
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

    if not args.output:
        sys.exit("Name of crops dataset must be provided.")
    elif not os.path.exists(os.path.join(RADIO_CROPS_PATH, args.output)):
        os.makedirs(os.path.join(RADIO_CROPS_PATH, args.output))


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

    parser.add_argument('-o', '--output', type=str,
                        help='Name of crops dataset that will be generated.')

    parser.add_argument('--histo', type=str, default=None,
                        help='Path to file with histo created' +
                        ' with "create_histo" script.')

    parser.add_argument('--fmt', type=str, default='dicom',
                        help='Format of input data. Default is "dicom".')

    parser.add_argument('--batch_size', type=int,
                        default=CONFIG.create_crops.batch_size,
                        help='Size of batch containing scans.' +
                        ' Default is {}.'.format(CONFIG
                                                 .create_crops
                                                 .batch_size))

    parser.add_argument('--rate', type=float,
                        default=CONFIG.create_crops.rate,
                        help='Expected number of cancerous' +
                        ' and non-cancerous crops per scan.' +
                        ' Default is {}.'.format(CONFIG
                                                 .create_crops
                                                 .rate))

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
                                                   .unify_spacing
                                                   .padding))

    args = parser.parse_args()

    # Checking passed arguments
    check_args(args)

    config_update = {
        'create_crops': {
            'rate': args.rate,
            'batch_size': args.batch_size
        },
        'preprocessing': {
            'mask_mode': args.mask_mode,
            'unify_spacing': {
                'padding': (args.padding_mode
                            if args.padding_mode != "zeros"
                            else "constant")
            }
        }
    }

    scans_index = ds.FilesIndex(path=args.dataset,
                                dirs=(args.fmt != 'raw'),
                                no_ext=(args.fmt == 'raw'))
    scans_dataset = ds.Dataset(scans_index, batch_class=CTImagesMaskedBatch)

    create_crops(dataset=scans_dataset, fmt=args.fmt,
                 nodules=pd.read_csv(args.annotation),
                 histo=load_histo(args.histo) if args.histo else None,
                 output_path=os.path.join(RADIO_CROPS_PATH, args.output),
                 config=config_update)
