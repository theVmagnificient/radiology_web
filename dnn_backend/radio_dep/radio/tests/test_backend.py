""" Contains tests for RadIO backend pipelines. """

import os
import shutil
import numpy as np
import pandas as pd
import pytest

from radio.preprocessing import CTImagesMaskedBatch
from radio.preprocessing import CTImagesBatch
from radio.dataset import Dataset, FilesIndex, Pipeline, F, B, C, V

from radio.pipelines.backend import accumulate_histo, create_crops, get_estimator
from radio.utils import get_config
from radio.utils import load_histo


config = {
    'preprocessing': {
        'unify_spacing': {
            'shape': (128, 128, 128),
            'spacing': (1.7, 1.0, 1.0),
            'order': 3
        },
        'mask_mode': 'ellipsoid'
    }
}


@pytest.mark.parametrize('bins', [8, (4, 4, 2), [8, 8, 4]])
def test_accamulate_histo(dicom_dataset, nodules, bins):
    accumulate_histo(dicom_dataset, 'dicom', nodules,
                     './histo.pkl', get_config(config))
    histo = load_histo('./histo.pkl')
    assert np.sum(histo[0]) > get_config(config).accumulate_histo.num_samples
    os.remove('./histo.pkl')


@pytest.mark.parametrize("input_shape", [(32, 64, 64), (16, 32, 32)])
@pytest.mark.parametrize("model_name", ['UNet', 'ResNet18'])
def test_get_estimator(model_name, input_shape):
    estimator = get_estimator(model_name,
                              './save_folder',
                              None, input_shape,
                              get_config(config))

    model = estimator.model
    _ = model.forward(model.as_torch(np.random.rand(2, *model.input_shape)))
    estimator.log_model()
    shutil.rmtree('./save_folder')


def test_create_crops(dicom_dataset, nodules):
    create_crops(dicom_dataset, 'dicom', nodules, None,
                 './test_crops', config=get_config(config))

    cancer_idx = FilesIndex(path='./test_crops/original/cancer/*', dirs=True)
    ncancer_idx = FilesIndex(path='./test_crops/original/ncancer/*', dirs=True)

    cancer_set = Dataset(cancer_idx, batch_class=CTImagesMaskedBatch)
    ncancer_set = Dataset(ncancer_idx, batch_class=CTImagesMaskedBatch)

    assert len(cancer_set) != 0 and len(ncancer_set) != 0

    _ = (
        Pipeline(dataset=cancer_set)
        .load(fmt='blosc', sync=True)
        .next_batch(2)
    )

    _ = (
        Pipeline(dataset=ncancer_set)
        .load(fmt='blosc', sync=True)
        .next_batch(2)
    )

    shutil.rmtree('./test_crops')
