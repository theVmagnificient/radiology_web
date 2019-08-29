import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import hashlib
import time

def get_nodules_pixel_coords(batch):
    """ get numpy array of nodules-locations and diameter in relative coords
    """
    nodules_dict = dict()
    nodules_dict.update(numeric_ix=batch.nodules.patient_pos)
    pixel_zyx = np.rint(abs((batch.nodules.nodule_center - batch.nodules.origin)) / batch.nodules.spacing).astype(np.int)
    nodules_dict.update({'coord' + letter: pixel_zyx[:, i] for i, letter in enumerate(['Z', 'Y', 'X'])})
    nodules_dict.update({'diameter_pixels': (np.rint(batch.nodules.nodule_size / batch.nodules.spacing).max()
                                             .astype(np.int))})
    pixel_nodules_df = pd.DataFrame.from_dict(nodules_dict).loc[:, ('numeric_ix', 'coordZ', 'coordY',
                                                                    'coordX', 'diameter_pixels')]
    return pixel_nodules_df


def dump_slices(batch_crops, save_path):
    df = get_nodules_pixel_coords(batch_crops)

    try:
        os.mkdir(os.path.join(save_path, batch_crops.indices[0]))
    except Exception as e:
        print(str(e))

    for slice_num in range(batch_crops.get(0, 'images').shape[0]):
        img = batch_crops.get(0, 'images')[slice_num]
        new_img = np.stack((img,) * 3, axis=-1)

        if slice_num in df['coordZ'].unique():
            for rep, nod in enumerate(df.loc[df.coordZ == slice_num].iterrows()):
                first_p = (
                int(nod[1].coordX - nod[1].diameter_pixels / 2.0), int(nod[1].coordY - nod[1].diameter_pixels / 2.0))
                second_p = (
                int(nod[1].coordX + nod[1].diameter_pixels / 2.0), int(nod[1].coordY + nod[1].diameter_pixels / 2.0))
                new_img = cv2.rectangle(new_img, first_p, second_p, (255, 0, 0), 2)

        new_img /= 255.0
        path = os.path.join(os.path.join(save_path, batch_crops.indices[0]), str(slice_num) + '.png')
        plt.imsave(path, new_img)


def check_overlap(batch_crops):
    l = []
    flag_overlap = False
    for i in range(len(batch_crops.nodules)):
        flag_overlap = False
        for j in range(len(batch_crops.nodules)):
            if i != j:
                vec = abs(batch_crops.nodules[i]['nodule_center'] - batch_crops.nodules[j]['nodule_center'])
                if np.sqrt(vec.dot(vec)) * 1.1 < float(
                        (batch_crops.nodules[i]['nodule_size'] + batch_crops.nodules[j]['nodule_size']).max() / 2.0):
                    if batch_crops.nodules[i]['nodule_size'].max() > batch_crops.nodules[j]['nodule_size'].max():
                        if batch_crops.nodules[j]['nodule_size'].max() < 8:
                            batch_crops.nodules[i]['nodule_size'] += batch_crops.nodules[j]['nodule_size']
                        l.append(batch_crops.nodules[i])
                    elif (batch_crops.nodules[i].nodule_size / batch_crops.nodules[i].spacing).max().astype(
                            np.int) > 11:
                        l.append(batch_crops.nodules[i])
                    flag_overlap = True
                    break

        if not flag_overlap:
            l.append(batch_crops.nodules[i])
    a = np.recarray((len(l),), dtype=batch_crops.data.nodules.dtype)
    for i in range(len(l)):
        a[i] = l[i]
    batch_crops.nodules = a
    print("Overlapping nodules processed")


def process_nodules(batch_crops):
    l = []
    for i in range(len(batch_crops.nodules)):
        if np.rint(batch_crops.nodules[i].nodule_size / batch_crops.nodules[i].spacing).max().astype(np.int) > 4:
            l.append(batch_crops.nodules[i])
    a = np.recarray((len(l),), dtype=batch_crops.data.nodules.dtype)
    for i in range(len(l)):
        a[i] = l[i]
    batch_crops.nodules = a
    batch_crops.masks = None
    batch_crops.create_mask()
    print("Crops processed")


def get_random_hash():
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    hash.hexdigest()
    return hash.hexdigest()[:10]




