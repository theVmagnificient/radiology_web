"""
Module with auxillary
    jit-compiled functions
    for lungs' segmentation
    from DICOM-scans
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from numba import jit
from skimage import measure, morphology


@jit(nogil=True)
def largest_label_volume(image, background=-1):
    """ Determine most frequent color that occupies the largest volume

    Parameters
    ----------
    image : ndarray
    background : int or float
        image background color

    Returns
    -------
    int
        most frequent color value
    """
    vals, counts = np.unique(image, return_counts=True)

    counts = counts[vals != background]
    vals = vals[vals != background]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return -999999


# segmentation of a patient sliced from skyscraper
@jit('void(double[:,:,:], double[:,:,:], double[:,:,:], int64)', nogil=True)
def calc_lung_mask_numba(patient, out_patient, res, erosion_radius=7):
    """ Compute lungs-segmenting mask for a patient

    Parameters
    ----------
    patient : ndarray
        input 3D scan.
    out_patient : ndarray
        resulting array with segmented lungs
    erosion_radius : int
        radius to use to erod the lungs' border
    res : ndarray
        `skyscraper` where to put the resized patient

    Returns
    -------
    tuple
        (res, out_patient.shape), resulting `skyscraper` and shape of
        segmented scan inside this `scyscraper`.

    """

    # slice the patient out from the skyscraper
    # we use view for simplification
    data = patient

    binary_image = np.array(data > -320, dtype=np.int8) + 1

    # 3d connected components
    labels = measure.label(binary_image)

    bin_shape = binary_image.shape

    # create np.array of unique labels of corner pixels
    corner_labs = np.zeros(0)

    for z_ind in range(bin_shape[0]):
        corner_labs = np.append(corner_labs, labels[z_ind, 0, 0])
        corner_labs = np.append(
            corner_labs, labels[z_ind, bin_shape[1] - 1, 0])
        corner_labs = np.append(
            corner_labs, labels[z_ind, bin_shape[1] - 1, bin_shape[2] - 1])
        corner_labs = np.append(
            corner_labs, labels[z_ind, 0, bin_shape[2] - 1])

    bk_labs = np.unique(corner_labs)

    # Fill the air around the person
    for background_label in bk_labs:
        binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    # For every slice we determine the largest solid structure

    # seems like to this point binary image is what it should be
    # return binary_image

    for i, axial_slice in enumerate(binary_image):

        axial_slice = axial_slice - 1

        # in each axial slice lungs and air are labelled as 0
        # everything else has label = 1

        # look for and enumerate connected components in axial_slice

        # 2d connected components
        labeling = measure.label(axial_slice)

        l_max = largest_label_volume(labeling, background=0)

        if l_max is not None:  # This slice contains some lung
            binary_image[i][labeling != l_max] = 1

    # return binary_image
    # it's all fine here

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body

    # stil fine
    # return binary_image

    # again, 3d connected components
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, background=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    # slightly erode the mask
    # to get rid of lungs' boundaries

    # return binary_image

    selem = morphology.disk(erosion_radius)

    # put the mask into the result
    for i in range(patient.shape[0]):
        out_patient[i, :, :] = morphology.binary_erosion(
            binary_image[i, :, :], selem)

    return res, out_patient.shape
