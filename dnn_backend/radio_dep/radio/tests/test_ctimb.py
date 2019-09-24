""" Contains tests for CTImagesMaskedBatch action-methods. """

import shutil
import pytest
import numpy as np
import pandas as pd

from radio import CTImagesMaskedBatch
from radio.dataset import FilesIndex
from radio.dataset import Dataset
from radio.dataset import Pipeline
from radio.models.utils import overlap_nodules


@pytest.fixture(scope='module')
def batch_gen(dicom_dataset):
    dicom_index = FilesIndex(path='./dicom/*', dirs=True)
    dicom_dataset = Dataset(dicom_index, batch_class=CTImagesMaskedBatch)

    create_blosc_dataset = dicom_dataset >> (
        Pipeline()
        .load(fmt='dicom')
        .dump(dst='./blosc', fmt='blosc',
              components=('images', 'origin', 'spacing'))
    )
    create_blosc_dataset.run(4)
    blosc_index = FilesIndex(path='./blosc/*', dirs=True)
    blosc_dataset = Dataset(blosc_index, batch_class=CTImagesMaskedBatch)
    yield blosc_dataset.gen_batch(2, n_epochs=None)
    print("Cleaning up generated blosc data...")
    shutil.rmtree('./blosc')


@pytest.fixture(scope='function')
def batch(batch_gen):
    return next(batch_gen).load(fmt='blosc', components=('images',
                                                         'origin',
                                                         'spacing'))


@pytest.fixture(scope='function')
def batch_with_nodules(batch, nodules):
    return batch.fetch_nodules_info(nodules=nodules, update=True)


@pytest.fixture(scope='function')
def batch_with_nodules_and_masks(batch_with_nodules):
    return batch_with_nodules.create_mask(mode='ellipsoid')


@pytest.fixture(scope='module', params=['constant', 'reflect', 'edge'])
def padding(request):
    return request.param


@pytest.fixture(scope='function')
def list_of_batches(batch_gen, batch_size=2, num_batches=2):
    batches = []
    for i in range(num_batches):
        batches.append(next(batch_gen))
    return batches


class TestCTIMB:

    def test_loaded_all(self, batch):
        assert len(np.nonzero(batch.images)) > 0
        assert batch.spacing.shape == (len(batch), 3)
        assert batch.origin.shape == (len(batch), 3)

    def test_fetch_nodules_info(self, batch, nodules):
        batch = batch.fetch_nodules_info(nodules=nodules)
        assert batch.nodules is not None
        assert (len(batch.nodules) ==
                batch.nodules_to_df(batch.nodules).shape[0])
        assert (batch.num_nodules ==
                batch.nodules_to_df(batch.nodules).shape[0])

    @pytest.mark.parametrize('mode', ['cuboid', 'ellipsoid'])
    def test_create_mask(self, batch, nodules, mode):
        batch = batch.fetch_nodules_info(nodules=nodules, update=True)
        batch = batch.create_mask(mode=mode)
        assert batch.masks is not None
        assert all(len(c) > 0 for c in np.nonzero(batch.masks))

    @pytest.mark.parametrize('shape,method,order', [((32, 64, 64), 'pil-simd', 2),
                                                    ((128, 24, 16), 'scipy', 3),
                                                    ((98, 91, 93), 'scipy', 5),
                                                    ((32, 32, 32), 'scipy', 1)])
    def test_resize(self, batch_with_nodules_and_masks, shape, method, order):
        batch = batch_with_nodules_and_masks.resize(shape=shape,
                                                    method=method,
                                                    order=order)
        assert np.all(batch.images_shape == np.array(shape, dtype=np.int))

    @pytest.mark.parametrize('order,spacing', [(1, (1.7, 1.0, 1.0)),
                                               (3, (0.5, 0.5, 0.1)),
                                               (5, (0.25, 0.25, 2.0))])
    def test_unify_spacing_no_mask(self, batch, order, spacing):

        batch = batch.unify_spacing(shape=(64, 64, 64), spacing=spacing,
                                    order=order, padding='reflect')
        assert np.all(batch.images_shape == np.array([64, 64, 64]))

    @pytest.mark.parametrize('shape,spacing', [((200, 200, 100), (1.7, 1.0, 1.0)),
                                               ((32, 32, 64), (0.5, 1.5, 0.1))])
    def test_unify_spacing_no_mask_padding(self, batch, shape, spacing, padding):
        batch = batch.unify_spacing(shape=shape, spacing=spacing,
                                    padding=padding, method='pil-simd')

        assert np.all(batch.images_shape == np.array(shape))

    @pytest.mark.parametrize('angle,axes', [(0.0, (0, 1)),
                                            (-15, (1, 2)),
                                            (45, (0, 2)),
                                            (-60, (2, 1)),
                                            (90, (1, 0)),
                                            (-120, (0, 1)),
                                            (180, (1, 2)),
                                            (-360, (2, 0))])
    def test_rotate(self, batch_with_nodules, angle, axes):
        _ = batch_with_nodules.rotate(angle=angle, axes=axes, random=True)  # noqa: F841
        _ = batch_with_nodules.rotate(angle=angle, axes=axes, random=False)  # noqa: F841

    def test_dicom_dump(self, batch_with_nodules_and_masks):
        _ = batch_with_nodules_and_masks.dump(dst='./dumped_dicoms',  # noqa: F841
                                              fmt='dicom')
        dicom_index = FilesIndex(path='./dumped_dicoms/*', dirs=True)
        dicom_dataset = Dataset(dicom_index, batch_class=CTImagesMaskedBatch)
        assert len(dicom_dataset) == len(batch_with_nodules_and_masks)

        batch = (  # noqa: F841
            dicom_dataset
            .next_batch(len(batch_with_nodules_and_masks))
            .load(fmt='dicom')
        )

        shutil.rmtree('./dumped_dicoms')

    @pytest.mark.parametrize('sync', [True, False])
    def test_blosc_dump_sync(self, batch_with_nodules_and_masks, sync):
        _ = batch_with_nodules_and_masks.dump(dst='./dumped_blosc',  # noqa: F841
                                              fmt='blosc', sync=sync)
        blosc_index = FilesIndex(path='./dumped_blosc/*', dirs=True)
        blosc_dataset = Dataset(blosc_index, batch_class=CTImagesMaskedBatch)
        assert len(blosc_dataset) == len(batch_with_nodules_and_masks)

        batch = (  # noqa: F841
            blosc_dataset
            .next_batch(len(batch_with_nodules_and_masks))
            .load(fmt='blosc', sync=sync)
        )

        shutil.rmtree('./dumped_blosc')

    @pytest.mark.parametrize('crop_size', [(16, 15, 17), (5, 30, 20)])
    def test_central_crop_inplace(self, batch_with_nodules_and_masks, crop_size):
        new_batch = batch_with_nodules_and_masks.central_crop(crop_size,
                                                              inplace=True,
                                                              crop_mask=True)
        assert batch_with_nodules_and_masks == new_batch
        assert np.allclose(new_batch.images_shape, crop_size)

    @pytest.mark.parametrize('crop_size', [(16, 15, 17), (5, 30, 20)])
    def test_central_crop(self, batch_with_nodules_and_masks, crop_size):
        new_batch = batch_with_nodules_and_masks.central_crop(crop_size,
                                                              inplace=False,
                                                              crop_mask=True)
        assert np.allclose(new_batch.images_shape, crop_size)
        assert not np.allclose(
            batch_with_nodules_and_masks.images_shape, crop_size)

    @pytest.mark.parametrize('crop_mask', [True, False])
    def test_central_crop_no_mask(self, batch_with_nodules,
                                  crop_mask, crop_size=(16, 16, 16)):
        new_batch = batch_with_nodules.central_crop(crop_size, inplace=False,
                                                    crop_mask=crop_mask)
        assert np.allclose(new_batch.images_shape, crop_size)
        assert not np.allclose(batch_with_nodules.images_shape, crop_size)
        assert new_batch.masks is None

    @pytest.mark.parametrize('threshold', [0.0, 0.3, 0.8, 1.0])
    def test_threshold_mask(self, batch_with_nodules_and_masks, threshold):
        new_batch = batch_with_nodules_and_masks.threshold_mask(
            threshold=threshold)
        assert id(new_batch) == id(batch_with_nodules_and_masks)

    @pytest.mark.parametrize('threshold', [0.0, 0.3, 0.8, 1.0])
    def test_binarize_mask(self, batch_with_nodules_and_masks, threshold):
        new_batch = batch_with_nodules_and_masks.binarize_mask(
            threshold=threshold)
        assert id(new_batch) == id(batch_with_nodules_and_masks)
        assert new_batch.masks.dtype in (
            int, np.int, np.uint8, np.int16, np.int32, np.int64)

    @pytest.mark.parametrize('threshold', [-2, 10, None, object()])
    def test_threshold_mask_bad_threshold(self, batch_with_nodules_and_masks, threshold):
        with pytest.raises((ValueError, TypeError)):
            _ = batch_with_nodules_and_masks.threshold_mask(  # noqa: F841
                threshold=threshold)

    @pytest.mark.parametrize('threshold', [-2, 10, None, object()])
    def test_binarize_mask_bad_threshold(self, batch_with_nodules_and_masks, threshold):
        with pytest.raises((ValueError, TypeError)):
            _ = batch_with_nodules_and_masks.binarize_mask(threshold=threshold)  # noqa: F841

    @pytest.mark.parametrize('stride', [(1, 2, 2), [2, 1, 2], np.array([2, 2, 1])])
    @pytest.mark.parametrize('patch_shape', [(16, 11, 13), [16, 16, 16], np.array([16, 16, 16])])
    @pytest.mark.parametrize('padding', ['edge', 'reflect'])
    def test_get_patches_images(self, batch, patch_shape, stride, padding):
        batch = batch.resize(shape=(64, 64, 64), order=3)
        _ = batch.get_patches(patch_shape, stride, padding, 'images')  # noqa: F841

    def test_split(self, batch_with_nodules_and_masks, batch_size=1):
        x, y = batch_with_nodules_and_masks.split(
            batch_with_nodules_and_masks, batch_size)
        if batch_size == 0:
            assert x is None and len(y) > 0
        elif batch_size >= len(batch_with_nodules_and_masks):
            assert len(x) == len(batch_with_nodules_and_masks) and y is None

    def test_concat(self, list_of_batches):
        concated_batch = CTImagesMaskedBatch.concat(list_of_batches)
        batches = [
            batch for batch in list_of_batches if batch is not None and len(batch) > 0]
        if len(batches) == 0:
            assert concated_batch is None
        else:
            assert len(concated_batch) == sum(len(batch) for batch in batches)
            for batch in batches:
                num_intersected = len(np.intersect1d(
                    concated_batch.indices, batch.indices))
                assert num_intersected == len(batch)

    @pytest.mark.parametrize('batch_size,crop_size,share', [(4, (1, 32, 32), 1.0),
                                                            (4, (16, 16, 16), 0.0),
                                                            (None, (16, 15, 1), 1.0),
                                                            (4, (16, 2, 16), 0.0),
                                                            (4, (16, 2, 16), 0.5)])
    def test_sample_nodules_no_mask(self, batch_with_nodules, batch_size, crop_size, share):
        x = batch_with_nodules.sample_nodules(batch_size, crop_size, share)
        if batch_size:
            assert len(x) == batch_size
        assert np.all(x.images_shape == np.array(crop_size))
        assert x.nodules is not None
        assert x.masks is None

    @pytest.mark.parametrize('batch_size,crop_size,share', [(4, (1, 32, 32), 1.0),
                                                            (4, (16, 16, 16), 0.0),
                                                            (None, (16, 15, 1), 1.0),
                                                            (4, (16, 2, 16), 0.0),
                                                            (4, (16, 2, 16), 0.5)])
    def test_sample_nodules_with_mask(self, batch_with_nodules_and_masks,
                                      batch_size, crop_size, share):
        x = batch_with_nodules_and_masks.sample_nodules(batch_size,
                                                        crop_size,
                                                        share)
        if batch_size:
            assert len(x) == batch_size
        assert np.all(x.images_shape == np.array(crop_size))
        assert np.all(x.masks_shape == np.array(crop_size))
        assert x.nodules is not None
        assert x.masks is not None

    def test_fetch_nodules_from_mask(self, batch_with_nodules_and_masks):
        batch = batch_with_nodules_and_masks
        old_nodules = batch.nodules

        batch = batch.fetch_nodules_from_mask()
        new_nodules = batch.nodules

        stats = overlap_nodules(batch, old_nodules, new_nodules)

        old_stats = pd.concat(
            [df for df in stats['true_stats'] if len(df) > 0])
        new_stats = pd.concat(
            [df for df in stats['pred_stats'] if len(df) > 0])

        assert old_stats.overlap_index.count() / old_stats.shape[0] > 0.8
        assert new_stats.overlap_index.count() / new_stats.shape[0] > 0.8


def test_dicom_dataset_non_empty(dicom_dataset):
    assert len(dicom_dataset) != 0


def test_dicom_load_all(dicom_dataset, batch_size=4):
    dicom_dataset.reset_iter()
    batch = dicom_dataset.next_batch(batch_size)

    batch = batch.load(fmt='dicom')
    assert len(np.nonzero(batch.images)) > 0
    assert batch.spacing.shape == (batch_size, 3)
    assert batch.origin.shape == (batch_size, 3)
