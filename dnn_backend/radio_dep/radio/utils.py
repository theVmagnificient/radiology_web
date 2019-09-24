""" Contains functions that can be helpful for initial CT-dataset manipulations. """

import os
import yaml
from binascii import hexlify
from os.path import dirname
import logging
import logging.config
import pickle
from pkg_resources import resource_filename
import numpy as np
from numba import njit


ENV_NAMES = [
    'RADIO_PATH',
    'RADIO_DATASETS_PATH',
    'RADIO_CROPS_PATH',
    'RADIO_PRETRAINED_PATH',
    'RADIO_EXPERIMENTS_PATH',
    'RADIO_LOGS_PATH'
]


class DotDict(dict):

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        value = self[key]
        return DotDict(value) if isinstance(value, dict) else value


def merge_configs(x, y):
    x, y = DotDict(x), DotDict(y)
    z = DotDict()
    for k in x.keys() | y.keys():
        if k not in y:
            z[k] = x[k]
        elif k not in x:
            z[k] = y[k]
        elif isinstance(y[k], dict) and isinstance(x[k], dict):
            z[k] = merge_configs(x[k], y[k])
        elif not isinstance(y[k], dict):
            z[k] = y[k]
        elif not isinstance(x[k], dict):
            z[k] = x[k]
        else:
            raise ValueError("Something went wrong!")
    return z


def get_config(config):
    return merge_configs(DEFAULT_CONFIG, config)


def str_to_tuple(input_str, dtype):
    values = (
        input_str
        .strip()
        .lstrip('(')
        .lstrip('[')
        .rstrip(')')
        .rstrip(']')
        .split(',')
    )
    return tuple(dtype(v) for v in values if len(v) > 0)


def generate_index(size=20):
    """ Generate random string index of givne size.

    Parameters
    ----------
    size : int
        length of index string.

    Returns
    -------
        string index of given size.
    """
    return hexlify(np.random.rand(100))[:size].decode()


def save_histo(histo, path):
    """ Save numpy histogram using pickle protocol.

    Parameters
    ----------
    histo : tuple(ndarray, ndarray)
        histogramm and bins edges which are both outputs of
        numpy.histogramdd function.
    path : str
        path to file where to save histogramm.
    """
    with open(path, 'wb') as file:
        pickle.dump(histo, file)


def load_histo(path):
    """ Load numpy histogram using pickle protocol.

    Parameters
    ----------
    path : str
        path to binary file containing pickled tuple
        with histogram and bins edges both represented
        by numpy ndarray.

    Returns
    tuple(ndarray, ndarray)
        tuple of histogramm and bins edges.
    """
    with open(path, 'rb') as file:
        histo = pickle.load(file)
    return histo


def get_initial_histo(scan_size, bins, num_samples=100):
    """ Get initial histogram.

    Parameters
    ----------
    scan_size : ArrayLike[int]
        size of scan.
    bins : ArrayLike[int]
        number of bins for each dimension.
        Note that (z,y,x)-ordering is used.
    num_samples : int
        number of random samples that will be used for
        histogram initialization. Default is 100.

    Returns
    -------
    histogram : NDArray
        The multidimensional histogram of sample x.
        See normed and weights for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.
    """
    scan_size = np.array(scan_size, np.int)
    data = np.random.rand(num_samples, 3) * scan_size
    ranges = np.stack([np.zeros(3, np.int), scan_size]).T
    return list(np.histogramdd(data, range=ranges, bins=bins))


#@njit
def get_malignancy(start, end, mask):
    num_nodules = start.shape[0]
    malignancy = np.zeros(num_nodules)
    for i in range(num_nodules):
        crop = mask[start[i, 0]: end[i, 0],
                    start[i, 1]: end[i, 1],
                    start[i, 2]: end[i, 2]]
        malignancy[i] = np.rint(np.mean(crop))
    return malignancy


def estimate_malignancy(batch, mal_mask):
    """ Estimate malignancy array from mask."""
    center_pix = np.abs(batch.nodules.nodule_center -
                        batch.nodules.origin) / batch.nodules.spacing
    start_pix = np.rint(batch.nodules.offset + (center_pix - \
                                        np.rint(batch.nodules.nodule_size /
                                                batch.nodules.spacing / 2))).astype(np.int64)
    end_pix = np.rint(start_pix + np.rint(batch.nodules.nodule_size / \
                                  batch.nodules.spacing)).astype(np.int64)

    start_pix[np.where(start_pix < 0)] = 0

    return np.rint(get_malignancy(start_pix, end_pix, mal_mask)).astype(np.int64)


def get_nodules_with_malignancy(batch, mal_mask):
    nodules = (
        batch
        .nodules_to_df(batch.nodules)
        .assign(malignancy=estimate_malignancy(batch, mal_mask))
    )
    return nodules


def fetch_tuple_from_str(input_str: str, dtype=float):
    """ Fetch tuple containing int or float values from input string. """
    elements = (
        input_str
        .lstrip('(')
        .lstrip('[')
        .rstrip(')')
        .rstrip(']')
        .split(',')
    )
    return tuple(dtype(e.strip()) for e in elements)


with open(resource_filename('radio', 'config/config.yml'), 'r') as file:
    try:
        DEFAULT_CONFIG = DotDict(yaml.load(file))

        for name in ENV_NAMES:
            value = os.environ.get(name, None)
            if value is not None:
                DEFAULT_CONFIG.GLOBAL[name] = value

        logs_path = DEFAULT_CONFIG.GLOBAL['RADIO_LOGS_PATH']
        print(logs_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        handlers = {}
        for handler_name, handler_config in DEFAULT_CONFIG.logging.handlers.items():
            if 'filename' in handler_config:
                update_value = handler_config['filename'].format(
                    RADIO_LOGS_PATH=logs_path)
                handlers[handler_name] = {**handler_config,
                                          'filename': update_value}
            else:
                handlers[handler_name] = handler_config
        DEFAULT_CONFIG['logging']['handlers'] = handlers
        logging.config.dictConfig(DEFAULT_CONFIG.logging)
    except yaml.YAMLError as err:
        logging.getLogger('RadIO').error(err)
