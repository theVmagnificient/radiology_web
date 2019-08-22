# pylint: disable=undefined-variable

""" Auxiliarry async functions for encoding and dump of data """

import os
import dill as pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import aiofiles
import blosc
from sklearn.cluster import MiniBatchKMeans
import pydicom
from pydicom import write_file, read_file
from pydicom import Dataset, FileDataset
from pydicom.multival import MultiValue


KMEANS_MINIBATCH = 10000
KMEANS_ITERS = 5


def dump_dicom(data, folder, spacing=(1, 1, 1),
               origin=(0, 0, 0), intercept=0, slope=1):
    """ Dump 3D scan in dicom format.

    Parameters
    ----------
    data : ndarray
        3D numpy array containing ct scan's data.
    folder : str
        folder where dicom files will be dumped.
    spacing : ArrayLike
        ndarray of shape (3,) that contains spacing along z, y, x axes.
    origin : ArrayLike
        ndarray of shape (3,) that contains origin for z, y, x axes.
    interception : float
        interception value. Default is 0.
    slope : float
        slope value. Default is 1.
    """
    spacing = np.array(spacing).reshape(-1)
    origin = np.array(origin).reshape(-1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    num_slices = data.shape[0]
    scan_id = np.random.randint(2 ** 16)
    for i in range(num_slices):
        slice_name = (
            hex(scan_id + i)
            .replace('x', '')
            .upper()
            .zfill(8)
        )
        filename = os.path.join(folder, slice_name)
        pixel_array = (data[i, ...] - intercept) / slope
        locZ, locY, locX = (float(origin[0] + spacing[0] * i),
                            float(origin[1]), float(origin[2]))

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = (
            hex(scan_id)
            .replace('x', '')
            .upper()
            .zfill(8)
        )
        file_meta.ImplementationClassUID = slice_name

        dataset = FileDataset(filename, {},
                              file_meta=file_meta,
                              preamble=b"\0"*128)

        dataset.PixelData = pixel_array.astype(np.uint16).tostring()
        dataset.RescaleSlope = slope
        dataset.RescaleIntercept = intercept

        dataset.ImagePositionPatient = MultiValue(type_constructor=float,
                                                  iterable=[locZ, locY, locX])

        dataset.PixelSpacing = MultiValue(type_constructor=float,
                                          iterable=[float(spacing[1]),
                                                    float(spacing[2])])
        dataset.SliceThickness = float(spacing[0])

        dataset.Modality = 'WSD'
        dataset.Columns = pixel_array.shape[0]
        dataset.Rows = pixel_array.shape[1]
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        dataset.PixelRepresentation = 1
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.SamplesPerPixel = 1

        write_file(filename, dataset)


def get_linear(from_interval, to_interval):
    """ Get linear transformation that maps one interval to another

    Parameters
    ----------
    from_interval : ndarray, tuple or list
        sequence of len=2 (llim, ulim) that defines the domain-interval
    to_interval : ndarray, tuple or list
        sequence of len=2 that defines the image-interval

    Returns
    -------
    function
        linear transformation
    """
    # compute coeffs of the mapping
    llim, ulim = from_interval
    new_llim, new_ulim = to_interval
    slope = (new_ulim - new_llim) / (ulim - llim)
    intercept = new_llim - slope * llim

    # define the map
    def linear(x):
        """ Transformation
        """
        return slope * x + intercept

    return linear


async def encode_dump_array_async(data, folder, filename, mode):
    """ Encode an ndarray to int8, blosc-pack it and dump data along with
    the decoder and shape of data into supplied folder.

    Parameters
    ----------
    data : ndarray
        contains numeric (e.g., float32) data to be dumped
    folder : str
        folder for dump
    filename : str
        name of file in which the data is dumped; has format name.ext
    mode : str or None
        Mode of encoding to int8. Can be either 'quantization' or 'linear'
        or None

    Notes
    -----
    currently, two modes of encoding are supported:
     - 'linear': maps linearly data-range to int8-range and then rounds off fractional part.
     - 'quantization': attempts to use histogram of pixel densities to come up with a
        transformation to int8-range that yields lesser error than linear mapping.
    """
    # parse mode of encoding
    if isinstance(mode, int):
        if mode <= 2:
            _modes = [None, 'linear', 'quantization']
            mode = _modes[mode]
    elif isinstance(mode, str):
        mode = mode.lower()

    fname_noext = '.'.join(filename.split('.')[:-1])

    # init list of serialized objects and filenames for dump
    byted, fnames = list(), list()

    # encode the data and get the decoder
    if mode == 'linear':
        data_range = (data.min(), data.max())
        i8_range = (-128, 127)

        if data_range[0] == data_range[1]:
            value = data_range[0]
            encoded = np.zeros_like(data, dtype=np.int8)

            def decoder(x): return x + value
        else:
            encoded = np.rint(get_linear(data_range, i8_range)
                              (data)).astype(np.int8)
            decoder = get_linear(i8_range, data_range)

        # serialize decoder
        byted.append(pickle.dumps(decoder))
        fnames.append(fname_noext + '.decoder')
    elif mode == 'quantization':

        # set up quantization model
        data_range = (data.min(), data.max())
        batch_size = min(KMEANS_MINIBATCH, data.size)
        model = MiniBatchKMeans(n_clusters=256, init=np.linspace(
            *data_range, 256).reshape(-1, 1))      # pylint: disable=no-member

        # fit the model on several minibatches, get encoded data
        for _ in range(KMEANS_ITERS):
            batch = np.random.choice(
                data.reshape(-1), batch_size, replace=False).reshape(-1, 1)
            model.partial_fit(batch)

        encoded = (model.predict(data.reshape(-1, 1)) - 128).astype(np.int8)

        # prepare decoder
        def decoder(x): return (model.cluster_centers_[
            x + 128]).reshape(data.shape)

        # serialize decoder
        byted.append(pickle.dumps(decoder))
        fnames.append(fname_noext + '.decoder')
    elif mode is None:
        encoded = data

    else:
        raise ValueError('Unknown mode of int8-encoding')

    # serialize (possibly) encoded data and its shape
    byted.extend([blosc.pack_array(encoded, cname='zstd',
                                   clevel=1), pickle.dumps(np.array(data.shape))])
    fnames.extend([filename, fname_noext + '.shape'])

    # dump serialized items
    for btd, fname in zip(byted, fnames):
        async with aiofiles.open(os.path.join(folder, fname), mode='wb') as file:
            _ = await file.write(btd)


def encode_dump_array_sync(data, folder, filename, mode):
    """ Encode an ndarray to int8, blosc-pack it and dump data along with
    the decoder and shape of data into supplied folder.

    Parameters
    ----------
    data : ndarray
        contains numeric (e.g., float32) data to be dumped
    folder : str
        folder for dump
    filename : str
        name of file in which the data is dumped; has format name.ext
    mode : str or None
        Mode of encoding to int8. Can be either 'quantization' or 'linear'
        or None

    Notes
    -----
    currently, two modes of encoding are supported:
     - 'linear': maps linearly data-range to int8-range and then rounds off fractional part.
     - 'quantization': attempts to use histogram of pixel densities to come up with a
        transformation to int8-range that yields lesser error than linear mapping.
    """
    # parse mode of encoding
    if isinstance(mode, int):
        if mode <= 2:
            _modes = [None, 'linear', 'quantization']
            mode = _modes[mode]
    elif isinstance(mode, str):
        mode = mode.lower()

    fname_noext = '.'.join(filename.split('.')[:-1])

    # init list of serialized objects and filenames for dump
    byted, fnames = list(), list()

    # encode the data and get the decoder
    if mode == 'linear':
        data_range = (data.min(), data.max())
        i8_range = (-128, 127)

        if data_range[0] == data_range[1]:
            value = data_range[0]
            encoded = np.zeros_like(data, dtype=np.int8)

            def decoder(x): return x + value
        else:
            encoded = np.rint(get_linear(data_range, i8_range)
                              (data)).astype(np.int8)
            decoder = get_linear(i8_range, data_range)

        # serialize decoder
        byted.append(pickle.dumps(decoder))
        fnames.append(fname_noext + '.decoder')
    elif mode == 'quantization':

        # set up quantization model
        data_range = (data.min(), data.max())
        batch_size = min(KMEANS_MINIBATCH, data.size)
        model = MiniBatchKMeans(n_clusters=256, init=np.linspace(
            *data_range, 256).reshape(-1, 1))      # pylint: disable=no-member

        # fit the model on several minibatches, get encoded data
        for _ in range(KMEANS_ITERS):
            batch = np.random.choice(
                data.reshape(-1), batch_size, replace=False).reshape(-1, 1)
            model.partial_fit(batch)

        encoded = (model.predict(data.reshape(-1, 1)) - 128).astype(np.int8)

        # prepare decoder
        def decoder(x): return (model.cluster_centers_[
            x + 128]).reshape(data.shape)

        # serialize decoder
        byted.append(pickle.dumps(decoder))
        fnames.append(fname_noext + '.decoder')
    elif mode is None:
        encoded = data

    else:
        raise ValueError('Unknown mode of int8-encoding')

    # serialize (possibly) encoded data and its shape
    byted.extend([blosc.pack_array(encoded, cname='zstd',
                                   clevel=1), pickle.dumps(np.array(data.shape))])
    fnames.extend([filename, fname_noext + '.shape'])

    # dump serialized items
    for btd, fname in zip(byted, fnames):
        with open(os.path.join(folder, fname), mode='wb') as file:
            file.write(btd)


def dump_data_sync(data_items, folder, i8_encoding_mode):
    """ Dump data from data_items on disk in specified folder

    Parameters
    ----------
    data_items : dict
        dict of data items for dump in form {item_name: [item, 'ext']}
        (e.g.: {'images': [scans, 'blk'], 'masks': [masks, 'blk'], 'spacing': [spacing, 'pkl']})
    folder : str
        folder to dump data-items in. Note that each data item is dumped in its separate subfolder
        inside the supplied folder.
    i8_encoding_mode: str, int, or dict
        contains mode of encoding to int8

    Notes
    -----
    Depending on supplied format in data_items, each data-item will be either
        pickle-serialized (if 'pkl') or blosc-packed (if 'blk')
    """

    # create directory if does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # infer extension of each item, serialize/blosc-pack and dump the item
    for item_name, (data, ext) in data_items.items():
        item_folder = os.path.join(folder, item_name)
        if not os.path.exists(item_folder):
            os.makedirs(item_folder)
        if ext == 'blk':
            if isinstance(i8_encoding_mode, dict):
                mode = i8_encoding_mode.get(item_name, None)
            else:
                mode = i8_encoding_mode

            encode_dump_array_sync(data, item_folder, 'data.blk', mode)

        elif ext == 'pkl':
            byted = pickle.dumps(data)
            with open(os.path.join(item_folder, 'data.pkl'), mode='wb') as file:
                file.write(byted)

    return None


async def dump_data_async(data_items, folder, i8_encoding_mode):
    """ Dump data from data_items on disk in specified folder

    Parameters
    ----------
    data_items : dict
        dict of data items for dump in form {item_name: [item, 'ext']}
        (e.g.: {'images': [scans, 'blk'], 'masks': [masks, 'blk'], 'spacing': [spacing, 'pkl']})
    folder : str
        folder to dump data-items in. Note that each data item is dumped in its separate subfolder
        inside the supplied folder.
    i8_encoding_mode: str, int, or dict
        contains mode of encoding to int8

    Notes
    -----
    Depending on supplied format in data_items, each data-item will be either
        pickle-serialized (if 'pkl') or blosc-packed (if 'blk')
    """

    # create directory if does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # infer extension of each item, serialize/blosc-pack and dump the item
    for item_name, (data, ext) in data_items.items():
        item_folder = os.path.join(folder, item_name)
        if not os.path.exists(item_folder):
            os.makedirs(item_folder)
        if ext == 'blk':
            if isinstance(i8_encoding_mode, dict):
                mode = i8_encoding_mode.get(item_name, None)
            else:
                mode = i8_encoding_mode

            _ = await encode_dump_array_async(data, item_folder, 'data.blk', mode)

        elif ext == 'pkl':
            byted = pickle.dumps(data)
            async with aiofiles.open(os.path.join(item_folder, 'data.pkl'), mode='wb') as file:
                _ = await file.write(byted)

    return None
