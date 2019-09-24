""" Contains functions that adapt pytorch models for RadIO cancer detection task. """

import os
import logging
import torch
from tqdm import tqdm
from toolz import curry
import numpy as np

from models.pytorch import models
from models.pytorch import Estimator
from models.pytorch.losses import dice_loss
from models.pytorch.metrics import tpr, fpr, precision, recall, fscore

from ..utils import get_config, save_histo, get_initial_histo
from ..utils import get_nodules_with_malignancy
from ..dataset import Dataset, Pipeline, FilesIndex
from ..preprocessing import CTImagesMaskedBatch
from ..named_expr import V, B, F
from .pipelines import combine_datasets, sample_simple as get_sample_dump_pipeline


def _dice_loss(y_pred, y_true):
    return dice_loss(y_true, y_pred)


def softmax(x):
    """ Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@curry
def _validate_classifier(batch: "CTImagesMaskedBatch",
                         estimator: "Estimator",
                         metrics: "Tuple[callable]",
                         crop_shape: "Tuple[int]" = (32, 64, 64),
                         strides: "Tuple[int]" = (32, 64, 64),
                         padding: str = "constant",
                         threshold: int = 10,
                         crops_batch_size: int = 4):
    """ Validate estimator of classification pytorch model on batch of scans.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch containing scans.
    estimator : Estimator
        estimator wrapper for pretrained classification
        pytorch model.
    metrics : Tuple[callable] or List[callable]
        metrics to compute via estimator.compute_metrics method.
    crop_shape : Tuple[int] or List[int]
        shape of crop along (z, y, x) axes. Default is (32, 64, 64).
    strides : Tuple[int] or List[int]
        strides for splitting on patches process. Default is (32, 64, 64).
        Lower values for strides along each axis lead to increased
        accuracy, but also significantly increase amount of
        computations needed to perform by GPU.
    padding : str
        type of padding. Can be "zeros", "constant" or "reflect".
        Default is "constant".
    threshold : int
        threshold for amount of cancerous pixels in crop
        to consider it cancerous.
    crop_batch_size : int
        size of batch containing crops. Default is 4.
    """
    inputs = batch.get_patches(patch_shape=crop_shape, stride=strides,
                               padding=padding, data_attr='images')
    inputs = inputs[:, np.newaxis, ...]
    masks = batch.get_patches(patch_shape=crop_shape, stride=strides,
                              padding=padding, data_attr='masks')
    targets = np.asarray([masks[i].sum() > threshold
                          for i in range(len(masks))], dtype=np.int)
    estimator.compute_metrics(inputs, targets, metrics, crops_batch_size,
                              store_labels_and_predictions=True)


def get_estimator(model_name: str,
                  save_folder: str,
                  device: int = 0,
                  input_shape: tuple = (32, 64, 64),
                  config: dict = {}) -> 'Estimator':
    """ Get estimator for standard adapted pytorch models.

    Parameters
    ----------
    model_name : str
        name of model class.
    save_folder : str
        path to folder where experiment will be saved.
        Experiment includes net weights and metrics evaluation.
    device : int
        number of cuda device to use. Default is 0.
    input_shape: Tuple[int, int, int]
        shape of input crop. Default is (32, 64, 64).
    config: DotDict
        configuration update for radio framework. Default is {}
        meaning that default radio config will be used.

    Returns
    -------
    Estimator
        estimator for given model class
    """
    logger = logging.getLogger('RadIO.backend')
    config = get_config(config)
    model_class = getattr(models, model_name)

    if model_name in ('VGG11', 'VGG13', 'VGG16C', 'VGG16D', 'VGG19'):
        model_config = {'input_shape': (1, *input_shape),
                        **config.models.VGG}
        loss_fn = torch.nn.CrossEntropyLoss()
        is_classifier = True  # noqa: F841

    elif model_name in ('DenseNet121', 'DenseNet161',
                        'DenseNet169', 'DenseNet201'):
        model_config = {'input_shape': (1, *input_shape),
                        **config.models.DenseNet}
        loss_fn = torch.nn.CrossEntropyLoss()
        is_classifier = True  # noqa: F841

    elif model_name in ('ResNet18', 'ResNet34',
                        'ResNet50', 'ResNet101',
                        'ResNet152'):
        model_config = {'input_shape': (1, *input_shape),
                        **config.models.ResNet}
        loss_fn = torch.nn.CrossEntropyLoss()
        is_classifier = True  # noqa: F841

    elif model_name in ('UNet', 'VNet'):
        model_config = {'input_shape': (1, *input_shape),
                        **config.models[model_name]}
        loss_fn = _dice_loss
        is_classifier = False  # noqa: F841

    else:
        raise ValueError("Incorrect name of model class." +
                         " Available models are:" +
                         " DenseNet121, DenseNet161", +
                         " DenseNet169, DenseNet201", +
                         " ResNet18, ResNet34 ResNet50, ResNet101, ResNet152", +
                         " VGG11, VGG13, VGG16C, VGG16D, VGG19," +
                         " UNet and VNet. Got {}.".format(model_name))

    logger.debug("Model's class: {}".format(model_name))
    logger.debug("Model's config: {}".format(model_config))

    estimator = Estimator(model_class(model_config),
                          save_folder, cuda_device=device)

    logger.debug("Model was loaded on {} cuda device".format(device))

    estimator.compile(loss=loss_fn, optimizer=torch.optim.Adam)
    return estimator


def accumulate_histo(dataset: 'Dataset',
                     fmt: str,
                     nodules: 'pandas.DataFrame',
                     output_path: str,
                     config: dict = {}):
    """ Accumulate information about nodules' centers location in histogram.

    Parameters
    ----------
    dataset : dataset.Dataset
        input dataset.
    fmt : str
        format for scans. Can be 'blosc', 'dicom' or 'raw'. Default is 'dicom'.
    nodules : pandas.DataFrame
        dataframe containing information about nodules location.
    output_path : str
        path to file where result histogram will be saved.
    config : DotDict
        config for radio framework. Default is {}
        meaning that default radio config dictionary will
        be used.
    """
    logger = logging.getLogger("RadIO.backend.training")
    logger.info("=======================================")
    logger.debug("Getting config dictionary")
    config = get_config(config)

    logger.info("Creating initial histogram with random samples")
    histo = get_initial_histo(
        config.preprocessing.unify_spacing.shape,
        bins=config.accumulate_histo.bins,
        num_samples=config.accumulate_histo.num_samples,
    )
    logger.info("Initial histogram with random samples was created")

    logger.debug("Creating pipeline for histogram accumulation")
    pipeline = dataset >> (
        Pipeline()
        .init_variable('pbar', tqdm(total=len(dataset),
                                    desc='Scans processed'))
        .call(lambda b: logger.info("Loading scans with indices {}".format(b.indices)))
        .load(fmt=fmt)
        .call(lambda b: logger.info("Fetching info about nodules"))
        .fetch_nodules_info(nodules=nodules)
        .call(lambda b: logger.info("Performing resize and spacing unification"))
        .unify_spacing(
            **config.preprocessing.unify_spacing
        )
        .call(lambda b: logger.info("Updating histogram"))
        .update_nodules_histo(histo)
        .call(lambda b: logger.info("Updating progress bar"))
        .update_variable('pbar', B('size'), mode='u')
    )

    logger.debug("Running pipeline for histogram accumulation")
    pipeline.run(batch_size=config.accumulate_histo.batch_size)
    logger.debug("Closing progress bar")
    pipeline.get_variable('pbar').close()
    logger.debug("Saving histogram into file with path {}".format(output_path))
    save_histo(histo, output_path)
    logger.info("Finished histogram accumulation!")
    logger.info("=======================================")


def create_crops(dataset: 'Dataset',
                 fmt: str,
                 nodules: 'pandas.DataFrame',
                 histo: 'Tuple[histogram, edges]',
                 output_path: str,
                 config: dict = {}):
    """ Create dataset of crops using dataset of scans and annotation.

    Returns pipeline with input dataset of scans attached to it
    for dataset of crops generation using info about annotation,
    histogram of nodules distribution inside scan.

    Parameters
    ----------
    dataset : dataset.Dataset
        input dataset.
    fmt : str
        format for scans. Can be 'blosc', 'dicom' or 'raw'. Default is 'dicom'.
    nodules : pandas.DataFrame
        dataframe containing information about nodules location.
    histo : Tuple[histogram, edges]
        output of get_initial_histo function.
    output_path: str
        path to directory where dataset of crops will be saved.
    config : dict
        config update for radio framework. Default is {}
        meaning that default radio config dictionary will
        be used.
    """
    logger = logging.getLogger("RadIO.backend.training")
    logger.info("=======================================")
    logger.info("Getting config dictionary")
    config = get_config(config)

    logger.debug("Building pipeline for crops dataset creation")
    pipeline = dataset >> (
        Pipeline()
        .init_variable('pbar', tqdm(total=len(dataset),
                                    desc='Scans processed'))
        .call(lambda b: logger.info("Loading scans with indices {}".format(b.indices)))
        .load(fmt=fmt)
        .call(lambda b: logger.info("Fetching info about nodules"))
        .fetch_nodules_info(nodules=nodules)
        .call(lambda b: logger.info("Performing resize and spacing unification"))
        .unify_spacing(
            **config.preprocessing.unify_spacing
        )
        .call(lambda b: logger.info("Creating mask"))
        .create_mask(mode=config.preprocessing.mask_mode)
        .call(lambda b: logger.info("Performing nodules sampling"))
        .call(lambda b: b >> get_sample_dump_pipeline(
            histo=histo, dst=output_path,
            crop_size=config.create_crops.crop_size,
            variance=config.create_crops.variance,
            rate=config.create_crops.rate)
        )
        .call(lambda b: logger.info("Updating progress bar value"))
        .update_variable('pbar', B('size'), mode='u')
    )

    logger.debug("Running pipeline for crops dataset generation")
    pipeline.run(batch_size=config.create_crops.batch_size)
    logger.debug("Closing progress bar")
    pipeline.get_variable('pbar').close()
    logger.info("Finished crops creation!")
    logger.info("=======================================")


def train_model(cancer_path: str, ncancer_path: str,
                estimator: 'Estimator',
                task='classification',
                config: dict = {}):
    """ Train CNN model on datasets of cancerous and non-cancerous crops.

    Parameters
    ----------
    cancer_path : str
        paths to cancerous crops.
    ncancer_path : str
        paths to non-cancerous crops.
    estimator : Estimator
        estimator wrapper for pytorch model to be learnt.
    task : str
        can be 'classification' or 'segmentation'. Default is 'classification'.
    config : dict
        config update for radio framework. Default is {}
        meaning that default radio config dictionary will
        be used.
    """
    logger = logging.getLogger("RadIO.backend.training")
    logger.info("=======================================")
    logger.debug("Getting config dictionary")
    config = get_config(config)
    num_epochs = config.training.num_epochs
    steps_per_epoch = config.training.steps_per_epoch
    crops_batch_sizes = config.training.crops_batch_sizes

    logger.debug("Creating FilesIndex with " +
                 "cancerous crops: '{}'".format(cancer_path))
    cancer_idx = FilesIndex(path=cancer_path, dirs=True)

    logger.debug("Creating FilesIndex with " +
                 "non-cancerous crops: '{}'".format(ncancer_path))
    ncancer_idx = FilesIndex(path=ncancer_path, dirs=True)

    logger.debug("Building dataset objects for created indices")
    cancer_set = Dataset(cancer_idx, batch_class=CTImagesMaskedBatch)
    ncancer_set = Dataset(ncancer_idx, batch_class=CTImagesMaskedBatch)
    logger.debug("Datasets were successfuly created!")

    save_folder = os.path.join(estimator.save_folder, 'model.tar')

    total_crops = num_epochs * steps_per_epoch * sum(crops_batch_sizes)

    logger.info("Creating pipeline for loading cancerous" +
                " and non-cancerous nodules")

    logger.debug("Building pipeline for model training on crops dataset")
    pipeline = (
        combine_datasets(
            datasets=[cancer_set, ncancer_set],
            batch_sizes=crops_batch_sizes,
            pipeline=(
                Pipeline()
                .load(fmt='blosc', components=('images', 'masks'), sync=True)
                .normalize_hu()
            )
        )
        .init_variable('pbar', tqdm(total=total_crops,
                                    desc='Crops processed'))
        .call(lambda b: logger.info("Fitting CNN model on batch data"))
        .call(
            lambda b: estimator.fit(
                b.unpack('images', data_format='channels_first'),
                (b.unpack('classification_targets').reshape(-1)
                 if task == 'classification'
                 else b.unpack('masks', data_format='channels_first')),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs, log_model=config.training.log_model,
                metrics=([tpr, fpr, precision, recall, fscore]
                         if task == 'classification' else None)
            )
        )
        .call(lambda b: (logger.info("Saving CNN model after epoch training finished")
                         if estimator._iteration_count % steps_per_epoch == 0
                         else None))
        .call(lambda b: (torch.save(estimator.model, save_folder)
                         if estimator._iteration_count % steps_per_epoch == 0
                         else None))
        .call(lambda b: logger.info("Updating progress bar"))
        .update_variable('pbar', B('size'), mode='u')
    )

    logger.debug("Running pipeline for model training")
    pipeline.run()
    logger.info("Training process is finished!")
    logger.info("Closing progress bar")
    pipeline.get_variable('pbar').close()
    logger.info("=======================================")


def evaluate_model(dataset: 'Dataset',
                   fmt: str,
                   nodules: 'pandas.DataFrame',
                   model: 'torch.Module',
                   device: int,
                   save_path: str,
                   task: str = 'segmentation',
                   config: dict = {}):
    """ Run validation pipeline for pretrained pytorch models.

    Parameters
    ----------
    dataset : Dataset
        input dataset.
    fmt : str
        format of input scans. Can be 'dicom' or 'raw'.
    model : pytorch Model
        pretrained pytorch model.
    device : int
        number of device.
    save_path : str
        path to directory where metrics will be saved.
    task : str
        task associated with pretrained model.
        Can be "classification" or "segmentation".
        Default is "segmentation".
    config : dict
        config update for radio framework. Default is {}
        meaning that default radio config dictionary will
        be used.
    """
    logger = logging.getLogger('RadIO.backend.evaluation')
    logger.info("=======================================")
    logger.debug("Getting config dictionary")
    config = get_config(config)
    logger.debug("Creating tqdm progress bar")
    pbar = tqdm(total=len(dataset), desc="Scans processed")

    logger.debug("Building preprocessing pipeline")
    preprocessing_pipeline = (
        Pipeline()
        .init_variable('pbar', pbar)
        .call(lambda b: logger.info("Loading scans with indices {}".format(b.indices)))
        .load(fmt=fmt)
        .call(lambda b: logger.info("Fetching info about nodules"))
        .fetch_nodules_info(nodules=nodules)
        .call(lambda b: logger.info("Performing resize and spacing unification"))
        .unify_spacing(**config.preprocessing.unify_spacing)
        .call(lambda b: logger.info("Creating mask"))
        .create_mask(mode=config.preprocessing.mask_mode)
        .call(lambda b: logger.info("Performing normalization"))
        .normalize_hu()
        .update_variable('pbar', B('size'), mode='u')
        .call(lambda b: logger.info("Preprocessing on batch of scans is finished"))
    )

    if task == 'segmentation':
        logger.info("Creating estimator for model")
        evaluator = Estimator(model, save_path,
                              cuda_device=device,
                              loss_fn=_dice_loss)
        logger.debug("Building pipeline for segmentation model evaluation")
        validation_pipeline = (
            preprocessing_pipeline

            +

            Pipeline()
            .init_variable('nodules_true')
            .init_variable('nodules_pred')
            .update_variable('nodules_true', B('nodules'))
            .call(lambda b: logger.info("Getting predictions of segmentation model"))
            .predict_on_scan(
                model=evaluator.predict,
                **config.evaluation.predict_on_scan,
            )
            .call(lambda b: logger.info("Performing mask thresholding"))
            .threshold_mask(threhsold=config.evaluation.threshold)
            .call(lambda b: logger.info("Fetching nodules from mask"))
            .fetch_nodules_from_mask()
            .update_variable('nodules_pred', B('nodules'))
            .call(lambda b: logger.info("Evaluating segmentation metrics"))
            .call(evaluator.compare_nodules,
                  nodules_true=V('nodules_true'),
                  nodules_pred=V('nodules_pred'))
            .call(lambda b: logger.info("Segmentation metrics has been evaluated"))
        )
    elif task == 'classification':
        evaluator = Estimator(model, save_path,
                              cuda_device=device,
                              loss_fn=torch.nn.CrossEntropyLoss())
        validation_pipeline = (
            preprocessing_pipeline

            +

            Pipeline()
            .call(lambda b: logger.info("Evaluating classification metrics"))
            .call(_validate_classifier(
                estimator=evaluator,
                metrics=[tpr, fpr, precision,
                         recall, fscore],
                crop_shape=config.evaluation.predict_on_scan.crop_shape,
                strides=config.evaluation.predict_on_scan.strides,
                crops_batch_size=config.evaluation.predict_on_scan.batch_size)
            )
            .call(lambda b: logger.info("Classification metrics has been evaluated"))
        )

    scans_batch_size = config.evaluation.scans_batch_size
    logger.debug(
        "Running pipeline for pretrained CNN model test metrics evaluation")
    (dataset >> validation_pipeline).run(batch_size=scans_batch_size)
    logger.info("Metrics evaluation process is finished!")
    logger.info("Closing progress bar")
    pbar.close()
    logger.info("=========================================")


def predict_with_composition(paths: str,
                             classifier: "Estimator",
                             segmentator: "Estimator",
                             malignator: "Estimator",
                             fmt: str = 'dicom',
                             logger: 'logger or None' = None,
                             config: dict = {}) -> 'DataFrame':
    """ Get prediction using three Estimators.

    Returns
    -------
    DataFrame
        containing predicted annotation.
    """
    config = get_config(config)

    logger = (logging.getLogger('RadIO.backend')
              if logger is None else logger)

    logger.info("Format is {}".format(fmt))
    logger.info("Paths are: {}".format(paths))
    index = FilesIndex(path=paths, no_ext=(fmt == 'raw'),
                       dirs=(fmt in ('blosc', 'dicom')))
    dataset = Dataset(index, batch_class=CTImagesMaskedBatch)

    logger.info(paths)
    logger.info(index.indices)
    logger.info(dataset.indices)
    logger.info('Making prediction')

    preprocess_pipeline = (
        Pipeline()
        .init_variable('shape', {})
        .init_variable('origin', {})
        .init_variable('spacing', {})
        .call(lambda b: logger.info("Started preprocessing" +
                                    " of scans: {}".format(b.indices)))
        .load(fmt=fmt)
        .update_variable('shape', value=F(lambda b: dict(zip(b.indices, b.images_shape))), mode='u')
        .update_variable('origin', value=F(lambda b: dict(zip(b.indices, b.origin))), mode='u')
        .update_variable('spacing', value=F(lambda b: dict(zip(b.indices, b.spacing))), mode='u')
        .unify_spacing(**config.preprocessing.unify_spacing)
        .normalize_hu()
        .call(lambda b: logger.info("Spacing in batch: {}".format(dict(zip(b.indices, b.spacing)))))
        .call(lambda b: logger.info("Origin in batch: {}".format(dict(zip(b.indices, b.origin)))))
        .call(lambda b: logger.info("Images Shapes in batch: {}".format(dict(zip(b.indices,
                                                                                 b.images_shape)))))
        .call(lambda b: logger.info("Finished preprocessing" +
                                    " of scans..."))
    )

    logger.info("Created preprocessing part of pipeline")
    if classifier is None:
        predict_heatmap_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("No classification model is used"))
            .init_variable('heatmap', 1)
        )
    else:
        predict_heatmap_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("Getting predictions of classification model"))
            .init_variable('heatmap')
            .call(lambda b: logger.info("Created heatmap"))
            .predict_on_scan(
                # Or -1 ? this is a property of classification model: [0] is proba of cancer
                model=lambda x:
                torch.nn.functional.softmax(
                    torch.from_numpy(
                        classifier.predict(x)))[..., 1],
                **{**config.deploy.predict_on_scan,
                   'targets_mode': 'classification'}
            )
            .call(lambda b: logger.info("Get prediction on scan"))
            .update_variable('heatmap', B('masks'))
            .call(lambda b: logger.info("Got predictions of classification model"))
        )
    logger.info("Created classification part of pipeline...")

    if malignator is None:
        predict_malignancy_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("No malignnacy classification " +
                                        "model is used..."))
            .init_variable('malignancy_heatmap', 0)
            .init_variable('malignancy_nodules', [])
            .update_variable('malignancy_heatmap',
                             F(lambda b: np.ones_like(b.images)))
        )
    else:
        predict_malignancy_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("Getting predictions of malignnacy " +
                                        "classification model..."))
            .init_variable('malignancy_heatmap')
            .init_variable('malignancy_nodules', [])
            .call(lambda b: logger.info("Created heatmap..."))
            .predict_on_scan(
                # Or -1 ? this is a property of classification model: [0] is proba of cancer
                model=lambda x: (np.argmax(
                    torch.nn.functional.softmax(
                        torch.from_numpy(
                            malignator.predict(x))), axis=1) + 1),
                **{**config.deploy.predict_on_scan,
                   'targets_mode': 'classification'}
            )
            .call(lambda b: logger.info("Get prediction on scan..."))
            .update_variable('malignancy_heatmap', B('masks'))
            .call(lambda b: logger.info("Got predictions of malignnacy " +
                                        "classification model"))
        )
    logger.info("Created malignancy classification part of pipeline...")

    if segmentator is None:
        predict_masks_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("No segmentation model is used"))
            .load(masks=F(lambda b: np.ones_like(b.images)), fmt='ndarray')
        )
    else:
        predict_masks_pipeline = (
            Pipeline()
            .call(lambda b: logger.info("Getting predictions of segmentation model"))
            .predict_on_scan(
                model=segmentator.predict,
                **{**config.deploy.predict_on_scan,
                   'targets_mode': 'segmentation'}
            )
            .call(lambda b: logger.info("Got predictions of segmentation model"))
            .call(lambda b: logger.info("Binarizing predicted mask..."))
            .binarize_mask(threshold=config.deploy.mask_proba_threshold)
        )
    logger.info("Created segmentation part of pipeline")
    pipeline = dataset >> (
        preprocess_pipeline
        + predict_heatmap_pipeline
        + predict_malignancy_pipeline
        + predict_masks_pipeline
        + (
            Pipeline()
            .load(fmt='ndarray', masks=B('masks') * V('heatmap'))
            .call(lambda b: logger.info("Fetching nodules from binarized mask"))
            .call(lambda b: logger.info("Spacing in batch before " +
                                        "fetching nodules: {}".format(dict(zip(b.indices,
                                                                               b.spacing)))))
            .call(lambda b: logger.info("Origin in batch before " +
                                        "fetching nodules: {}".format(dict(zip(b.indices,
                                                                               b.origin)))))
            .call(lambda b: logger.info("Images Shapes in batch before " +
                                        "fetching nodules: {}".format(dict(zip(b.indices,
                                                                               b.images_shape)))))
            .fetch_nodules_from_mask()
            .update_variable('malignancy_nodules',
                             F(lambda b: get_nodules_with_malignancy(
                                 b, V('malignancy_heatmap').get(b))),
                             mode='a')
            .call(lambda b: logger.info("Complete!"))
        )
    )

    batch = pipeline.next_batch(1)

    malignancy_df = pipeline.get_variable('malignancy_nodules')[0]

    _shapes_dict = pipeline.get_variable('shape')
    _spacing_dict = pipeline.get_variable('spacing')
    _origin_dict = pipeline.get_variable('origin')

    _shape = tuple(int(v) for v in _shapes_dict[batch.indices[0]])
    _spacing = tuple(float(v) for v in _spacing_dict[batch.indices[0]])
    _origin = tuple(float(v) for v in _origin_dict[batch.indices[0]])
    logger.debug("Prediction was made...")
    return (
        batch
        .nodules_to_df(batch.nodules)
        .assign(malignancy=malignancy_df['malignancy'])
        .assign(IShapeZ=_shape[0], IShapeY=_shape[1], IShapeX=_shape[2])
        .assign(ISpacingZ=_spacing[0], ISpacingY=_spacing[1], ISpacingX=_spacing[2])
        .assign(IOriginZ=_origin[0], IOriginY=_origin[1], IOriginX=_origin[2])
        .rename(columns={
                'source_id': 'seriesuid',
                'locZ': 'coordZ',
                'locY': 'coordY',
                'locX': 'coordX'}
                )
        .query('(coordZ - IOriginZ) / ISpacingZ >= 0 and (coordZ - IOriginZ) / ISpacingZ <= IShapeZ')
        .query('(coordY - IOriginY) / ISpacingY >= 0 and (coordY - IOriginY) / ISpacingY <= IShapeY')
        .query('(coordX - IOriginX) / ISpacingX >= 0 and (coordX - IOriginX) / ISpacingX <= IShapeX')
    )
