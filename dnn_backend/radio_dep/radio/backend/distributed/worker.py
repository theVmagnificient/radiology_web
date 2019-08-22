#!/usr/bin/python
""" Contains implementation of worker class used in distributed RadIO framework. """

import os
import sys
from os.path import dirname
from functools import wraps
import time
import logging
import argparse
import torch
import Pyro4
from Pyro4.errors import ConnectionClosedError
from multiprocessing import Process

from models.pytorch import Estimator
from models.pytorch.losses import log_loss, dice_loss  # noqa: H301
from models.pytorch.models import UNet, VGG19  # noqa: H301
from radio.pipelines.backend import predict_with_composition
from radio.pipelines.backend import _dice_loss
from radio.backend.distributed.workitem import Workitem
from radio.dataset import FilesIndex
from pydicom import read_file
from radio.utils import DEFAULT_CONFIG as CONFIG
from radio.utils import merge_configs


Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')


HEARTBEAT_TIMEOUT = 0.3
SLEEP_TIME = 0.2
PYRO_CONNECTION_TIMEOUT = 30.5


RADIO_EXPERIMENTS_PATH = CONFIG.GLOBAL.RADIO_EXPERIMENTS_PATH
RADIO_PRETRAINED_PATH = CONFIG.GLOBAL.RADIO_PRETRAINED_PATH
RADIO_LOGS_PATH = CONFIG.GLOBAL.RADIO_LOGS_PATH


def reconnect(method):
    @wraps(method)
    def _decorated(self, *args, **kwargs):
        proxy = self.dispatcher
        while True:
            try:
                result = method(self, *args, **kwargs)
            except ConnectionClosedError:
                proxy._pyroReconnect()
                continue

            return result
    return _decorated


class RadIOWorker:

    def __init__(self, name,
                 segmentator=None,
                 classifier=None,
                 malignator=None,
                 device='gpu',
                 batch_size=8,
                 threshold=0.1,
                 heartbeat_interval=0.04):
        self.logger = logging.getLogger('RadIO.worker.' + name + '_heartbeat')
        if device == 'cpu':
            device = None
        elif device == 'gpu':
            device = 0
        else:
            device = int(device)

        self.logger.info("Device is set to '{}'".format(device))

        if segmentator is not None:
            self.logger.info("Getting estimator for segmenation model...")
            try:
                self.segmentator = Estimator(segmentator, '../sgm',
                                             loss_fn=_dice_loss,
                                             cuda_device=device)
            except Exception as e:
                self.logger.error("Error {} occured when building ".format(e) +
                                  " segmentation estimator.")
                sys.exit(1)
            self.logger.info("Successfully built estimator " +
                             " for segmentation model...")

        else:
            self.logger.info("No segmentation model will " +
                             "be used for prediction...")
            self.segmentator = None

        if classifier is not None:
            self.logger.info("Getting estimator for classification model...")
            try:
                self.classifier = Estimator(classifier, '../clf',
                                            loss_fn=torch.nn.CrossEntropyLoss(),
                                            cuda_device=device)
            except Exception as e:
                self.logger.error("Error {} occured when building ".format(e) +
                                  " classification estimator.")
                sys.exit(1)
            self.logger.info("Successfully built estimator " +
                             " for classification model...")
        else:
            self.logger.info("No classification model will " +
                             "be used for prediction...")
            self.classifier = None

        if malignator is not None:
            self.logger.info("Getting estimator for malignancy model...")

            try:
                self.malignator = Estimator(malignator, '../mlf',
                                            loss_fn=torch.nn.CrossEntropyLoss(),
                                            cuda_device=device)
            except Exception as e:
                self.logger.error("Error {} occured when building ".format(e) +
                                  " segmentation estimator.")
                sys.exit(1)
            self.logger.info("Successfully built estimator " +
                             " for malignancy model...")
        else:
            self.logger.info("No malignancy model will " +
                             "be used for prediction...")
            self.malignator = None

        self.batch_size = int(batch_size)
        self.device = device
        self.threshold = threshold

        self.worker_name = name
        self.heartbeat_interval = float(heartbeat_interval)

        self.logger.info("Worker name is {}".format(name))
        self.logger.info("Batch size is set to {}".format(self.batch_size))
        self.logger.info("Threshold is set to {}".format(self.threshold))
        self.logger.info("Heartbeat interval "
                         + "is set to {}".format(self.heartbeat_interval))

        self.logger.info("Getting dispatchers proxy...")
        self.dispatcher = Pyro4.core.Proxy('PYRONAME:dispatcher')

    @reconnect
    def get_work(self):
        return self.dispatcher.get_work(self.worker_name)

    @reconnect
    def put_result(self, item):
        return self.dispatcher.put_result(self.worker_name, item)

    @reconnect
    def send_heartbeat(self):
        while True:
            self.dispatcher.heartbeat(self.worker_name)
            time.sleep(self.heartbeat_interval)

    def _process_nodules(self,
                         nodules: 'DataFrame',
                         diameter_threshold: float = 5.0,
                         confidence_threshold: float = 0.0) -> 'DataFrame':
        self.logger.info(
            "Processing nodules: renaming columns and filtering...")
        return (
            nodules
            .assign(diameter_mm=lambda df: (df.loc[:, ['diamZ',
                                                       'diamY',
                                                       'diamX']] *
                                            df.loc[:, ['ISpacingZ',
                                                       'ISpacingY',
                                                       'ISpacingX']]).max(axis=1))
            .loc[:, ['seriesuid',
                     'coordX',
                     'coordY',
                     'coordZ',
                     'diameter_mm',
                     'nodule_id',
                     'confidence',
                     'malignancy',
                     'IShapeX',
                     'IShapeY',
                     'IShapeZ',
                     'ISpacingX',
                     'ISpacingY',
                     'ISpacingZ',
                     'IOriginX',
                     'IOriginY',
                     'IOriginZ']]
            .query("confidence > @confidence_threshold")
            .query("diameter_mm > @diameter_threshold")
        )

    def _check_item(self, item):
        if not isinstance(item, Workitem):
            raise TypeError("Request must be instance of Workitem." +
                            " Got {} instead.".format(type(item)))
        if not isinstance(item.scan_path, str):
            raise TypeError("Scan path must be str.")
        if item.fmt not in ('dicom', 'raw'):
            raise ValueError("Format must be 'dicom' or 'raw'.")

    def _check_scan(self, item):
        if item.fmt == 'raw':
            index = FilesIndex(path=item.scan_path,
                               no_ext=False, dirs=False)
            if len(index.indices) == 0:
                raise FileNotFoundError("File with given path does not exist")
            if any('.mhd' not in str(p) for p in index.indices):
                raise ValueError("File must have '.mhd' extension.")
        if item.fmt == 'dicom':
            if not (os.path.exists(item.scan_path) and os.path.isdir(item.scan_path)):
                raise FileNotFoundError("DICOM-directory " +
                                        "with given path does not exist.")
            for name in os.listdir(item.scan_path):
                path = os.path.join(item.scan_path, name)
                try:
                    _ = read_file(path)  # noqa: F841
                except Exception:
                    raise ValueError("Scans path must be" +
                                     " directory containing dicom files.")

    def _get_config(self, item):
        if isinstance(item.config, dict):
            _config_update = {
                'predict_on_scan': {
                    'strides': item.config.get('strides',
                                               CONFIG
                                               .deploy
                                               .predict_on_scan
                                               .strides),
                    'batch_size': item.config.get('batch_size',
                                                  CONFIG
                                                  .deploy
                                                  .predict_on_scan
                                                  .batch_size),
                },
                'mask_proba_threshold': item.config.get('mask_proba_threshold',
                                                        CONFIG
                                                        .deploy
                                                        .mask_proba_threshold)
            }
            return merge_configs(CONFIG, {'deploy': _config_update})
        else:
            return CONFIG

    def predict_on_scan(self, path, fmt='dicom', config=None) -> dict:
        config = CONFIG if config is None else config
        if config.deploy.use_classifier:
            classifier = self.classifier
        else:
            classifier = None

        if config.deploy.use_segmentator:
            segmentator = self.segmentator
        else:
            segmentator = None

        if config.deploy.use_malignator:
            malignator = self.malignator
        else:
            malignator = None

        nodules = predict_with_composition(
            paths=path, fmt=fmt,
            logger=self.logger,
            config=config,
            segmentator=segmentator,
            classifier=classifier,
            malignator=malignator
        )
        return self._process_nodules(nodules)

    def run(self):
        heartbeat_process = Process(target=self.send_heartbeat)
        heartbeat_process.start()
        self.logger.debug("Started heartbeat process...")
        while True:
            item = self.get_work()
            try:
                self._check_item(item)
                self._check_scan(item)
            except Exception as e:
                self.logger.debug("Exception occured during work" +
                                  "item's internal data validation: {}".format(e))
                item.predicted_nodules = e
                self.put_result(item)
                continue

            self.logger.debug("Got work item: {}".format(item))
            try:
                item.predicted_nodules = self.predict_on_scan(
                    item.scan_path, fmt=item.fmt,
                    config=self._get_config(item)
                )
            except Exception as e:
                self.logger.error("Exception occured during " +
                                  "prediction: {}".format(e))
                item.predicted_nodules = e
            self.logger.debug("Putting result: {}".format(item))
            self.put_result(item)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=(
        "This program runs one worker instance in separate process."
        " Each instance contains one ANN model that will be used for prediction.")
    )

    parser.add_argument('-n', '--name', type=str,
                        help='Name of worker instance.')

    parser.add_argument('-s', '--segmentator', type=str,
                        default=os.path.join(RADIO_PRETRAINED_PATH,
                                             'unet3d/'),
                        help=('Path to segmentation model' +
                              ' that will be used for prediction.'))

    parser.add_argument('-c', '--classifier', type=str,
                        default=os.path.join(RADIO_PRETRAINED_PATH,
                                             '/res50/'),
                        help=('Path to classification model' +
                              ' that will be used for prediction.'))

    parser.add_argument('-m', '--malignator', type=str,
                        default=os.path.join(RADIO_PRETRAINED_PATH,
                                             '/malignancy/'),
                        help=('Path to malignancy classification model' +
                              ' that will be used for prediction.'))

    parser.add_argument('--device', type=int, default=0,
                        help="Device number that will be used for prediction.")

    args = parser.parse_args()
    name = args.name
    segmentator_name, classifier_name, malignator_name = (args.segmentator,
                                                          args.classifier,
                                                          args.malignator)
    device = args.device
    # dispatcher_address, name = str(sys.argv[1]), str(sys.argv[2])
    logger = logging.getLogger('RadIO.worker.' + args.name + '_heartbeat')
    handler = logging.handlers.RotatingFileHandler(os.path.join(RADIO_LOGS_PATH,
                                                                name + '.log'),
                                                   maxBytes=5242880,
                                                   backupCount=1)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s " +
                                  "- %(threadName)s \n%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('Loading segmentation model....')

    if classifier_name is not None and not os.path.exists(classifier_name):
        classifier_name = os.path.join(RADIO_EXPERIMENTS_PATH,
                                       classifier_name,
                                       'model.tar')
    else:
        classifier_name = os.path.join(classifier_name, 'model.tar')

    if malignator_name is not None and not os.path.exists(malignator_name):
        malignator_name = os.path.join(RADIO_EXPERIMENTS_PATH,
                                       malignator_name,
                                       'model.tar')
    else:
        malignator_name = os.path.join(malignator_name, 'model.tar')

    if not os.path.exists(segmentator_name):
        segmentator_name = os.path.join(RADIO_EXPERIMENTS_PATH,
                                        segmentator_name,
                                        'model.tar')
    else:
        segmentator_name = os.path.join(segmentator_name, 'model.tar')

    if classifier_name is not None and not os.path.exists(classifier_name):
        logger.error("No saved classification model" +
                     " found with path: {}".format(classifier_name))
        sys.exit(1)

    if malignator_name is not None and not os.path.exists(malignator_name):
        logger.error("No saved classification model" +
                     " found with path: {}".format(malignator_name))
        sys.exit(1)

    if not os.path.exists(segmentator_name):
        logger.error("No saved segmentation model" +
                     " found with path: {}".format(segmentator_name))
        sys.exit(1)

    sgm_model = torch.load(segmentator_name, None).eval().train(False)
    logger.info('Model was successfully loaded!')

    if classifier_name is not None:
        clf_model = torch.load(classifier_name, None).eval().train(False)
    else:
        clf_model = None
    logger.info('Classification model was successfully loaded!')

    if malignator_name is not None:
        mlg_model = torch.load(malignator_name, None).eval().train(False)
    else:
        mlg_model = None
    logger.info('Malignancy classification model was successfully loaded!')

    logger.info("Starting worker...")
    worker = RadIOWorker(name,
                         sgm_model,
                         clf_model,
                         mlg_model,
                         device=device)
    worker.run()
