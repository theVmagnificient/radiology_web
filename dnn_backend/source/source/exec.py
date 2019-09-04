from pipeline import Pipe
from resnet import resnet34
from mod_unet import *
from torch.optim import Adam
from models.pytorch.losses import dice_loss
#from models.pytorch.estimator import Estimator
from new_estimator import NewEstimator
from config import Config

from unpacker import Preprocess
import os
import shutil


class Executor:

    def __init__(self, cf):
        print("Init started")
        unet = Modified3DUNet(1, 1)
        resnet = resnet34()
        device = torch.device('cpu')

        unet.load_state_dict(torch.load(cf.pathToSegmentator, map_location=device))
        resnet.load_state_dict(torch.load(cf.pathToClassifier, map_location=device))
        print("Models loaded")

        self.segmentator = NewEstimator(unet, save_folder='./experiments/unet_full_pipe_eval/',
                                     cuda_device="cpu",
                                     optimizer=Adam, loss_fn=dice_loss)
        self.classify = NewEstimator(resnet, save_folder='./experiments/res_full_pipe_eval/',
                                  cuda_device="cpu",
                                  optimizer=Adam, loss_fn=torch.nn.CrossEntropyLoss())
        print("Estimators created")

        self.pipe = Pipe(cf, self.classify, self.segmentator)

        try:
            shutil.rmtree(cf.save_path)
        except Exception as e:
            print(str(e))

        try:
            os.makedirs(cf.save_path)
            print("Directory created")
        except Exception as e:
            print(str(e))

        print("Init complited")

    def unpack(self, pathToArchive, pathToConverted, numWorkers=3):
        prep = Preprocess(pathToArchive, pathToConverted, numWorkers)
        prep.start()
        self.pipe.add_dataset(pathToConverted)

    def start(self):
        self.pipe.start_inference()


if __name__ == '__main__':
    conf = Config('/mnt/results/experiments')
    e = Executor(cf=conf)
    e.unpack('/mnt/archives/out.zip', '/mnt/archives/tmp')
    e.start()
