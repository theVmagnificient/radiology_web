import logging
import os
import numpy as np
import pydicom
import radio
from radio.preprocessing.ct_batch import AIR_HU, DARK_HU
from radio import dataset as ds
from radio import B, V, C
from radio import CTImagesMaskedBatch, CTImagesBatch
from radio.dataset import inbatch_parallel
import numpy as np
import pickle
import blosc

logger = logging.getLogger('dicom')
logger.setLevel(logging.DEBUG)


class CTImagesDicomBatch(CTImagesMaskedBatch):
    components = "images", "masks", "spacing", "origin", "nodules", 'origin_beg', 'spacing_beg', 'series'

    def __init__(self, index, *args, **kwargs):
        """ Execute Batch construction and init of basic attributes

        Parameters
        ----------
        index : Dataset.Index class.
            Required indexing of objects (files).
        """
        super().__init__(index, *args, **kwargs)
        self.masks = None
        self.nodules = None
        self.occupancy_grid = None
        # self.add_components(['origin_beg', 'spacing_beg'], [self.origin, self.spacing])

    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_dicom(self, patient_id, components=None, **kwargs):
        logger = logging.getLogger('dicom')
        print('loading_started')
        components = np.intersect1d(components, CTImagesBatch.components)
        if 'images' not in components:
            raise ValueError("Components argument must contain 'images'"
                             + " value if format is 'dicom' or 'raw'")
        patient_pos = self.index.get_pos(patient_id)
        patient_folder = self.index.get_fullpath(patient_id)
        print(patient_folder)
        dicom_names = [patient_folder + '/' + s for s in os.listdir(patient_folder)]
        list_of_dicoms = [pydicom.dcmread(patient_folder + '/' + s)
                          for s in os.listdir(patient_folder)]

        logger.warning("Sorting slices in accesending order...")

        print(len(list_of_dicoms))
        list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        dicom_slice = list_of_dicoms[0]
        intercept_pat = dicom_slice.RescaleIntercept
        slope_pat = dicom_slice.RescaleSlope

        z_spacing = sum(np.diff([x.ImagePositionPatient[2] for x in list_of_dicoms])) / len(list_of_dicoms)

        if 'spacing' in components:
            self.spacing[patient_pos, ...] = np.asarray([float(round(z_spacing, 2)),
                                                         float(dicom_slice.PixelSpacing[1]),
                                                         float(dicom_slice.PixelSpacing[0])], dtype=np.float)

        if 'origin' in components:
            self.origin[patient_pos, ...] = np.asarray([float(dicom_slice.ImagePositionPatient[2]),
                                                        float(dicom_slice.ImagePositionPatient[1]),
                                                        float(dicom_slice.ImagePositionPatient[0])], dtype=np.float)

        # self.add_components(['origin_beg', 'spacing_beg'], [self.origin, self.spacing])
        self.origin_beg = self.origin
        self.spacing_beg = self.spacing

        patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)
        print('Images stacked')

        patient_data[patient_data == AIR_HU] = 0

        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)

        self.series = int(dicom_slice.SeriesNumber)

        return patient_data

    def _load_blosc_component_sync(self, ix, component, ext):
        """ Load component stored in blosc format.

        Parameters
        ----------
        ix : str
            index of element in batch.
        component : str
            name of component.
        ext : str
            extension of file stored on hard drive.
        """

        component_path = os.path.join(self.index.get_fullpath(ix),
                                      component, 'data' + '.' + ext)

        if not os.path.exists(component_path):
            raise OSError("File with component "
                          + "{} doesn't exist.".format(component))

        with open(component_path, mode='rb') as file:
            byted = file.read()

        if ext == 'blk':
            decoder_path = os.path.join(self.index.get_fullpath(ix),
                                        component, 'data.decoder')

            if os.path.exists(decoder_path):
                with open(decoder_path, mode='rb') as file:
                    decoder = pickle.loads(file.read())
            else:
                decoder = lambda x: x

            data = decoder(blosc.unpack_array(byted))

        elif ext == 'pkl':
            data = pickle.loads(byted)

        # print(self.get_pos(None, component, ix))
        if component == 'spacing_beg':
            self.spacing_beg = data
            return
        if component == 'origin_beg':
            self.origin_beg = data
            return
        if component == 'series':
            self.series = data
            return
        component_pos = self.get_pos(None, component, ix)
        getattr(self, component)[component_pos] = data
