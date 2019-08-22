#!/usr/bin/python

import os
import sys
import argparse
from tqdm import tqdm
import glob
import pandas as pd

from radio import dataset as ds
from radio import CTImagesMaskedBatch
from radio.utils import get_config
from radio.utils import DEFAULT_CONFIG as CONFIG

RADIO_DATASETS_PATH = CONFIG.GLOBAL['RADIO_DATASETS_PATH']


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

    if args.fmt not in ('dicom', 'raw'):
        sys.exit("Incorrect format: must be 'raw' or 'dicom'")

    if args.mask_mode not in ('ellipsoid', 'cuboid'):
        sys.exit("Incorrect mask creation mode: must be 'ellipsoid' or 'cuboid'")

    if args.padding_mode not in ('zeros', 'reflect', 'edge', 'constant'):
        sys.exit("Argument 'padding_mode' must be 'zeros'," +
                 " 'constant', 'reflect' or 'edge'")

    if os.path.exists(args.annotation):
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
    elif not os.path.exists(os.path.join(DATASETS_DIR, args.output)):
        os.makedirs(os.path.join(DATASETS_DIR, args.output))


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

    parser.add_argument('--fmt', type=str, default='dicom',
                        help='Format of input data: "dicom" or "raw". Default is "dicom".')

    parser.add_argument('--batch_size', type=int,
                        default=CONFIG.create_crops.batch_size,
                        help='Size of batch containing scans.' +
                        ' Default is {}.'.format(CONFIG
                                                 .create_crops
                                                 .batch_size))

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
        'preprocessing': {
            'mask_mode': args.mask_mode,
            'unify_spacing': {
                'padding': (args.padding_mode
                            if args.padding_mode != "zeros"
                            else "constant")
            }
        }
    }

    config = get_config(config_update)
    scans_index = ds.FilesIndex(path=args.dataset,
                                dirs=(args.fmt != 'raw'),
                                no_ext=(args.fmt == 'raw'))
    scans_dataset = ds.Dataset(scans_index, batch_class=CTImagesMaskedBatch)
    pbar = tqdm(total=len(scans_dataset), desc='Scans processed')
    pipeline = (
        ds.Pipeline()
        .init_variable('pbar', pbar)
        .load(fmt=args.fmt)
        .unify_spacing(**config.preprocessing.unify_spacing)
    )

    if args.annotation:
        pipeline = (
            pipeline
            .fetch_nodules_info(nodules=pd.read_csv(args.annotation))
            .create_mask(mask_mode=config.preprocessing.mask_mode)
        )

    pipeline = (
        pipeline
        .dump(dst=os.path.join(RADIO_DATASETS_PATH, args.output),
              components=['origin', 'spacing', 'images'] +
              ['masks', 'nodules'] if args.annotation else [])
        .update_variable('pbar', B('size'), mode='u')
    )
    (scans_dataset >> pipeline).run(batch_size=args.batch_size)
    pbar.close()
