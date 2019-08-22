#!/usr/bin/python

import os
import sys
import argparse
import torch

from models.pytorch import models

from radio.pipelines.backend import get_estimator
from radio.pipelines.backend import train_model
from radio.utils import DEFAULT_CONFIG as CONFIG
from radio.utils import str_to_tuple


RADIO_CROPS_PATH = CONFIG.GLOBAL['RADIO_CROPS_PATH']
RADIO_EXPERIMENTS_PATH = CONFIG.GLOBAL['RADIO_EXPERIMENTS_PATH']


def check_args(args):

    if args.task == 'segmentation' and args.model not in ('UNet', 'VNet'):
        sys.exit("Segmentation task is only supported for UNet and VNet models.")

    try:
        _ = getattr(models, args.model)  # noqa: F841
    except KeyError:
        sys.exit("Incorrect name of model. See -m option.")
    except AttributeError:
        sys.exit("Incorrect name of model. See -m option.")
    except TypeError:
        sys.exit("Model name must be provided")

    if args.task not in ('classification', 'segmentation'):
        sys.exit("Task must be 'classification' or 'segmentation'. See -t option.")

    if not (args.steps > 0 and isinstance(args.steps, int)):
        sys.exit("Number of steps per epoch must be positive integer. See -s option.")

    if not (args.epochs > 0 and isinstance(args.epochs, int)):
        sys.exit("Number of epochs must be positive integer. See -e option.")

    try:
        batch_sizes = str_to_tuple(args.batch_sizes, int)
    except Exception:
        sys.exit("Batch sizes must be tuple of positive integers." +
                 " See --batch_sizes option.")
    if not (batch_sizes[0] > 0 and batch_sizes[1] > 0):
        sys.exit("Batch sizes must be tuple of positive integer." +
                 " See --batch_sizes option.")

    if args.device < 0 or args.device >= torch.cuda.device_count():
        sys.exit("Incorrect device id. Must be positive" +
                 " integer [0...{}]".format(torch.cuda.device_count() - 1) +
                 " See --device option.")

    if not args.dataset:
        sys.exit("Dataset path must be provided. See -d option.")
    elif not os.path.exists(os.path.join(RADIO_CROPS_PATH, args.dataset)):
        sys.exit("Incorrect path to dataset: {}".format(
            os.path.join(RADIO_CROPS_PATH, args.dataset)))
    elif not (os.path.exists(os.path.join(RADIO_CROPS_PATH, args.dataset, 'cancer/original/')),
              os.path.exists(os.path.join(RADIO_CROPS_PATH, args.dataset, 'ncancer/original/'))):
        sys.exit("Incorrect path to dataset")

    if not args.output:
        sys.exit("Name of experiment must be provided. See -o option.")
    elif not os.path.exists(os.path.join(RADIO_EXPERIMENTS_PATH, args.output)):
        os.makedirs(os.path.join(RADIO_EXPERIMENTS_PATH, args.output))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "This script allows to train classification" +
        " or segmentation model on dataset of crops.")
    )

    parser.add_argument('-d', '--dataset', type=str,
                        help=('Name of crops dataset that will be used for training.'))

    parser.add_argument('-m', '--model', type=str,
                        help='Name of model that will be trained.')

    parser.add_argument('-t', '--task', type=str, default='classification',
                        help='Type of task. Can be classification or segmentation')

    parser.add_argument('-s', '--steps', type=int,
                        default=CONFIG.training.steps_per_epoch,
                        help='Number of training steps per epoch.' +
                        'Default is {}.'.format(CONFIG
                                                .training
                                                .steps_per_epoch))

    parser.add_argument('-e', '--epochs', type=int,
                        default=CONFIG.training.num_epochs,
                        help='Number of training epochs.' +
                        ' Default is {}.'.format(CONFIG
                                                 .training
                                                 .num_epochs))

    parser.add_argument('-o', '--output', type=str,
                        help='Name of experiment.')

    parser.add_argument('--batch_sizes', type=str, default='(4, 4)',
                        help='Size of batch containing cropss. Default is (4, 4).')

    parser.add_argument('--device', type=int, default=0,
                        help=('GPU device number that will be used' +
                              ' for model training. Default is 0.'))

    args = parser.parse_args()

    # Checking passed arguments
    check_args(args)

    config_update = {
        "training": {
            "num_epochs": args.epochs,
            "steps_per_epoch": args.steps,
            "crops_batch_sizes": str_to_tuple(args.batch_sizes, int)
        }
    }

    train_model(
        cancer_path=os.path.join(RADIO_CROPS_PATH, args.dataset,
                                 'cancer/original/*'),
        ncancer_path=os.path.join(RADIO_CROPS_PATH, args.dataset,
                                  'ncancer/original/*'),
        estimator=get_estimator(model_name=args.model, device=args.device,
                                input_shape=CONFIG.create_crops.crop_size,
                                save_folder=os.path.join(RADIO_EXPERIMENTS_PATH,
                                                         args.output),
                                config=config_update),
        task=args.task, config=config_update
    )
