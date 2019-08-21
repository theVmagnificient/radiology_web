import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys, os, cv2

COLOR_MAPS = ("COLORMAP_BONE", "COLORMAP_JET", "COLORMAP_HOT")
cmaps = {"bone": plt.cm.bone, "gray": plt.cm.gray}

def export_to_png(path, dic_ds):
    # make it True if you want in PNG format
    image = ""
    # Specify the .dcm folder path
    for n, ds in enumerate(dic_ds):
        image = str(n) + "_{}.png"
        
        for cmName, cm in cmaps.items():
            fig = plt.figure(frameon=False, dpi=300)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(ds.pixel_array, cmap=cm)     
            plt.savefig(os.path.join(path, image.format(cmName)))
            plt.close()