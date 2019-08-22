""" Contains tests for RadIO pipelines. """

import shutil
import numpy as np
import pytest

from radio.preprocessing import CTImagesMaskedBatch
from radio.dataset import Dataset
from radio.dataset import FilesIndex
from radio.dataset import Pipeline
from radio.pipleines import combine_datasets
from radio.pipelines import split_dump
from radio.pipelines import update_histo


@pytest.fixture(scope='module')
def histo(dicom_dataset, nodules, sizes=(128, 128, 128), bins=4):
    sizes = np.array(sizes)
    init_data = np.random.uniform(0, 1, size=(100, 3))
    histo = np.histogramdd(init_data * sizes[np.newaxis, :],
                           range=list(zip([0, 0, 0], sizes)), bins=bins)
    histo = list(histo)
    _ = update_histo(nodules, histo, fmt='dicom', shape=(128, 128, 128),  # noqa: F841
                     spacing=(1.7, 1.0, 1.0), order=3, padding='reflect')
    return histo


@pytest.fixture(scope='module')
def crops_datasets(dicom_dataset, nodules, histo):
    pipeline = split_dump('./cancer', './ncancer', nodules, histo,
                          fmt='dicom', spacing=(1.7, 1.0, 1.0),
                          shape=(128, 128, 128), order=3,
                          padding='reflect', crop_size=(32, 64, 64))

    pipeline = (dicom_dataset >> pipeline)

    pipeline.next_batch(2)

    cancer_idx = FilesIndex(path='./cancer/*', dirs=True)
    ncancer_idx = FilesIndex(path='./ncancer/*', dirs=True)

    cancer_set = Dataset(cancer_idx, batch_class=CTImagesMaskedBatch)
    ncancer_set = Dataset(ncancer_idx, batch_class=CTImagesMaskedBatch)

    yield cancer_set, ncancer_set

    shutil.rmtree('./cancer')
    shutil.rmtree('./ncancer')


class TestRadioPipelines:

    def test_update_histo(self, dicom_dataset, nodules,
                          sizes=(128, 128, 128), bins=4):
        sizes = np.array(sizes)
        init_data = np.random.uniform(0, 1, size=(100, 3))
        histo = np.histogramdd(init_data * sizes[np.newaxis, :],
                               range=list(zip([0, 0, 0], sizes)), bins=bins)
        histo = list(histo)
        num_items_before = np.sum(histo[0])
        pipeline = update_histo(nodules, histo, fmt='dicom',
                                shape=sizes, spacing=(1.7, 1.0, 1.0),
                                order=3, padding='reflect')

        (dicom_dataset >> pipeline).run(2)
        num_items_after = np.sum(histo[0])
        assert num_items_before != num_items_after

    def test_split_dump(self, dicom_dataset, nodules, histo):
        pipeline = split_dump('./temp_cancer', './temp_ncancer',
                              nodules, histo, fmt='dicom',
                              spacing=(1.7, 1.0, 1.0),
                              shape=(128, 128, 128), order=3,
                              padding='reflect', crop_size=(32, 64, 64))

        pipeline = (dicom_dataset >> pipeline)

        pipeline.next_batch(2)
        pipeline.next_batch(2)

        cancer_idx = FilesIndex(path='./temp_cancer/*', dirs=True)
        ncancer_idx = FilesIndex(path='./temp_ncancer/*', dirs=True)

        assert len(cancer_idx) > 0
        assert len(ncancer_idx) > 0

        shutil.rmtree('./temp_cancer')
        shutil.rmtree('./temp_ncancer')

    @pytest.mark.parametrize('components', [('images', 'origin', 'spacing'),
                                            ('images', 'masks',
                                             'origin', 'spacing'),
                                            ('images', 'nodules',
                                             'origin', 'spacing'),
                                            ('images', 'nodules', 'masks',
                                             'origin', 'spacing')])
    @pytest.mark.parametrize('batch_sizes', [(2, 2), (1, 3), (0, 4)])
    def test_combine_datasets(self, crops_datasets, batch_sizes, components):
        pipeline = (
            Pipeline()
            .load(fmt='blosc', components=components)
            .normalize_hu()
        )
        combine_pipeline = combine_datasets(crops_datasets,
                                            batch_sizes,
                                            pipeline)
        _ = combine_pipeline.next_batch(4)  # noqa: F841
