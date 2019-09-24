""" Visualization utils functions """

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import interactive, widgets, HBox, VBox
from ipykernel.pylab.backend_inline import flush_figures


def plot_slices(batch, predict=None, select=None, size=20, grid=True):
    ''' Plot batch's image, mask, and prediction
         Parameters:
        -----------
        batch : radio.CTImagesMaskedBatch
            batch with batch.images and batch.masks loaded
        predict : numpy.array
            predicted segmentation mask, shpuld be of same size as image
            of dimensions either (batch_size, 1, z-dim, y-dim, x-dim) or
            (batch_size, z-dim, y-dim, x-dim, 1)
        select : list of int or None
            choose what figures to plot, providing theirs id (int) as list,
            see Note
        size : int or tuple of int
            specify total image size. If int, image is a square
        grid : bool
            Whether to apply grid on all figures or not

         Note:
        -----
        Predefined plots and ids:
        1:
        {'args': (img, plt.cm.bone)},
        2:
        {'args': (mask, plt.cm.bone)},
        3:
        {'args': (mask * img, plt.cm.bone)},
        4:
        {'args': (img + mask * 300, plt.cm.seismic)},
        5:
        {'args': (pred, plt.cm.bone)},
        6:
        {'args': (pred * img, plt.cm.bone)},
        7:
        {'args': (img + pred * 300, plt.cm.seismic)}
        }
    '''
    return plot_slices_matplotlib(batch, predict,
                                  select, size, grid)


def plot_slices_matplotlib(batch, predict=None, select=None,
                           size=20, grid=True):
    ''' Plot slices and mask with interact
    '''
    if isinstance(size, int):
        size1, size2 = size, size
    else:
        size1, size2 = size
    select = [1, 2, 3, 4, 5, 6, 7] if select is None else select
    indices = list(batch.indices)
    n_s = batch.images_shape[0][0]
    if predict is not None:
        try:
            predict = np.squeeze(predict, axis=1)
        except ValueError:
            predict = np.squeeze(predict, axis=-1)
    batch.fetch_nodules_from_mask()
    center_scaled = (np.abs(batch.nodules.nodule_center -
                            batch.nodules.origin) /
                     batch.nodules.spacing)
    nod_size_scaled = (np.rint(batch.nodules.nodule_size /
                               batch.nodules.spacing)).astype(np.int)
    nods = np.concatenate([batch.nodules.patient_pos.reshape(-1, 1),
                           center_scaled.astype(np.int),
                           nod_size_scaled[:, 0].reshape(-1, 1)], axis=1)
    nods = widgets.Dropdown(
        options=nods.tolist(),
        value=nods.tolist()[0],
        description='Nodule:',
    )
    slid = widgets.IntSlider(min=0, max=n_s, value=0)
    patient_id = widgets.Dropdown(
        options=indices,
        value=indices[0],
        description='Patient ID',
    )

    def update_loc(*args):
            slid.value = nods.value[1]
            patient_id.value = indices[nods.value[0]]
    nods.observe(update_loc, 'value')

    def upd(patient_id, n_slice, nods):
        img = batch.get(patient_id, 'images')[n_slice]
        mask = batch.get(patient_id, 'masks')[n_slice]
        nonlocal predict
        if predict is None:
            rows, cols = 1, len(select)
            pred = np.zeros_like(img)
            fig, axes = plt.subplots(rows, cols,
                                     squeeze=False, figsize=(size1, size2))
        else:
            rows = np.ceil(len(select) / 3).astype(np.int)
            cols = 3 if rows > 1 else len(select)
            fig, axes = plt.subplots(rows, cols,
                                     squeeze=False, figsize=(size1, size2))
            # where fun begins :D
            if 4 or 5 or 6 in select:
                    pred = predict[indices.index(patient_id), n_slice, ...]
        if grid:
            inv_spacing = 1 / batch.get(patient_id, 'spacing').reshape(-1)[1:]
            step_mult = 10
            xticks = np.arange(0, img.shape[0], step_mult * inv_spacing[0])
            yticks = np.arange(0, img.shape[1], step_mult * inv_spacing[1])
        all_plots = {1:
                     {'args': (img, plt.cm.bone)},
                     2:
                     {'args': (mask, plt.cm.bone)},
                     3:
                     {'args': (mask * img, plt.cm.bone)},
                     4:
                     {'args': (img + mask * 300, plt.cm.seismic)},
                     5:
                     {'args': (pred, plt.cm.bone)},
                     6:
                     {'args': (pred * img, plt.cm.bone)},
                     7:
                     {'args': (img + pred * 300, plt.cm.seismic)}
                     }
        i = 0
        for r in range(rows):
            for c in range(cols):
                axes[r][c].imshow(*all_plots[select[i]]['args'])
                axes[r][c].set_xticks(xticks, minor=True)
                axes[r][c].set_yticks(yticks, minor=True)
                axes[r][c].grid(color='r', linewidth=1.5,
                                alpha=0.15, which='minor')
                i += 1
                if i == len(select):
                    break
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=0.4,
                            wspace=0.1, hspace=0.1)
        flush_figures()
    w = interactive(upd, patient_id=patient_id, n_slice=slid, nods=nods)
    w.children = (VBox([HBox(w.children[:-1]), w.children[-1]]),)
    display(w)
