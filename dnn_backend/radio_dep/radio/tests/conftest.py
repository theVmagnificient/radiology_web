""" Contains different fixtures required by tests. """

import os
import shutil
import numpy as np
import pandas as pd
import pytest
import pydicom
from pydicom import write_file
from pydicom import Dataset as DicomDataset
from pydicom import FileDataset as DicomFileDataset
from pydicom.multival import MultiValue
from radio.dataset import FilesIndex
from radio.dataset import Dataset
from radio.dataset import Pipeline
from radio import F
from radio import CTImagesMaskedBatch


def generate_dicom_scans(dst, num_scans=10, intercept=0, slope=1):
    spacing = (0.4 + 0.4 * np.random.rand(num_scans, 3) +
               np.array([1 + 0.5 * np.random.rand(), 0, 0]))
    origin = np.random.randint(-200, 200, (num_scans, 3))
    for i in range(num_scans):
        num_slices = np.random.randint(128, 169)
        scan_id = np.random.randint(2 ** 16)
        scan_data = np.random.randint(0, 256, (num_slices, 128, 128))
        folder = os.path.join(
            dst, hex(scan_id).replace('x', '').upper().zfill(8))

        if not os.path.exists(folder):
            os.makedirs(folder)

        for k in range(num_slices):
            slice_name = (
                hex(scan_id + k)
                .replace('x', '')
                .upper()
                .zfill(8)
            )
            filename = os.path.join(folder, slice_name)
            pixel_array = (scan_data[k, ...] - intercept) / slope
            locZ = float(origin[i, 0] + spacing[i, 0] * k)
            locY, locX = float(origin[i, 1]), float(origin[i, 2])

            file_meta = DicomDataset()
            file_meta.MediaStorageSOPClassUID = "Secondary Capture Image Storage"
            file_meta.MediaStorateSOPInstanceUID = (
                hex(scan_id)
                .replace('x', '')
                .upper()
                .zfill(8)
            )

            file_meta.ImplementationClassUID = slice_name

            dataset = DicomFileDataset(filename, {},
                                       file_meta=file_meta,
                                       preamble=b"\0" * 128)

            dataset.PixelData = pixel_array.astype(np.uint16).tostring()
            dataset.RescaleSlope = slope
            dataset.RescaleIntercept = intercept

            dataset.ImagePositionPatient = MultiValue(type_constructor=float,
                                                      iterable=[locZ, locY, locX])

            dataset.PixelSpacing = MultiValue(type_constructor=float,
                                              iterable=[float(spacing[i, 1]),
                                                        float(spacing[i, 2])])
            dataset.SliceThickness = float(spacing[i, 0])

            dataset.Modality = 'WSD'
            dataset.Columns = pixel_array.shape[0]
            dataset.Rows = pixel_array.shape[1]
            dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            dataset.PixelRepresentation = 1
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.SamplesPerPixel = 1

            write_file(filename, dataset)


def generate_nodules(batch, diam_mean=16, nodules_per_scan=10):
    num_nodules = nodules_per_scan * len(batch)
    patients_pos = np.random.choice(len(batch), num_nodules)
    diams = np.random.chisquare(diam_mean, num_nodules)
    max_diams_pixels = diams / np.min(batch.spacing[patients_pos, :], axis=1)
    coords = (np.random.rand(num_nodules, 3) *
              batch.images_shape[patients_pos, :] -
              max_diams_pixels[:, np.newaxis])
    coords = (coords * batch.spacing[patients_pos, :] +
              diams[:, np.newaxis] + batch.origin[patients_pos, :])

    nodules_df = pd.DataFrame({
        'seriesuid': batch.indices[patients_pos],
        'coordZ': coords[:, 0],
        'coordY': coords[:, 1],
        'coordX': coords[:, 2],
        'diameter_mm': diams
    })
    return nodules_df


@pytest.fixture(scope='module')
def dicom_dataset():
    if os.path.exists('./dicom'):
        shutil.rmtree('./dicom')

    generate_dicom_scans('./dicom')

    index = FilesIndex(path='./dicom/*', dirs=True)
    dataset = Dataset(index, batch_class=CTImagesMaskedBatch)
    yield dataset
    print("Cleaning up generated dicom data...")
    shutil.rmtree('./dicom')


@pytest.fixture(scope='module')
def nodules(dicom_dataset):
    pipeline = dicom_dataset >> (
        Pipeline()
        .init_variable('nodules_list', [])
        .load(fmt='dicom')
        .update_variable('nodules_list', F(generate_nodules), mode='a')
    )
    pipeline.run(batch_size=2)

    all_nodules = pd.concat(
        [df for df in pipeline.get_variable('nodules_list') if len(df) > 0])
    return all_nodules
