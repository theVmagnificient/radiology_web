from radio.dataset import Pipeline
from radio.dataset import FilesIndex, Dataset
import torch
from dicom_batch import B, V, C
from dicom_batch import CTImagesDicomBatch
from radio import dataset as ds
import os
import pandas as pd
from utils import dump_slices, check_overlap, process_nodules, get_nodules_dict


class Pipe:

    def __init__(self, cf, classifier, segmentator):

        self.cf = cf
        self.full_pipe = (Pipeline()
            .init_variable('segm_mask')
            .init_variable('conf_mask')
            .load(fmt='dicom')
            .unify_spacing(shape=(500, 300, 300), spacing=(1.0, 1.0, 1.0),
                           method='pil-simd', padding='constant')
            #.call(crop_img)
            .normalize_hu()
            .predict_on_scan(
                model=lambda x:
                    torch.nn.functional.softmax(
                        torch.from_numpy(
                            classifier.predict(x)))[..., 1],
                    crop_shape=[64, 64, 64],
                    strides=[55, 55, 55],
                    batch_size=4,
                    data_format="channels_first",
                    model_type="callable",
                    targets_mode="classification",
                    show_progress=False
                )
            #.binarize_mask(threshold=0.7)
            .update_variable('conf_mask', B('masks'))
            .predict_on_scan(
                        model=segmentator.predict,
                        crop_shape=(64, 64, 64),
                        strides=(55, 55, 55),
                        batch_size=4,
                        data_format="channels_first",
                        model_type="callable",
                        targets_mode="segmentation",
                        show_progress=False
                    )
            .binarize_mask(threshold=0.1)
            .update_variable('segm_mask', B('masks'))
            .load(fmt='ndarray', masks=V('segm_mask') * V('conf_mask'))
            .fetch_nodules_from_mask()
            .call(check_overlap)
            .call(process_nodules))

    def add_dataset(self, path):
        dicom_ix = ds.FilesIndex(path=os.path.join(path, '*'), dirs=True)
        dicom_dataset = Dataset(index=dicom_ix, batch_class=CTImagesDicomBatch)

        print(f"Dataset length: {len(dicom_dataset)}")
        self.full_pipe = dicom_dataset >> self.full_pipe

    def start_inference(self):
        print("Started inference")
        nodules_dicts = []
        nods = pd.DataFrame(
            columns=["nodule_id", "source_id", "locZ", "locY", "locX", "diamZ", "diamY", "diamX", "confidence",
                     "series"])

        csv_path = os.path.join(self.cf.save_path, 'res.csv')
        nods.to_csv(csv_path, index=False)
        for dcm in range(len(self.full_pipe)):
            print(dcm)
            batch_crops = self.full_pipe.next_batch(1, shuffle=False)
            dump_slices(batch_crops, self.cf.save_path)
            nodules_dicts.append(get_nodules_dict(batch_crops))
            a = batch_crops.nodules_to_df(batch_crops.nodules)
            a.to_csv(csv_path, mode='a', header=False, index=False)
        return nodules_dicts




