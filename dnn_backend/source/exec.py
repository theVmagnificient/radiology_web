from pipeline import Pipe
from resnet import resnet34
from mod_unet import *
from torch.optim import Adam
from models.pytorch.losses import dice_loss
#from models.pytorch.estimator import Estimator
from new_estimator import NewEstimator as Estimator
from config import Config

from unpacker import Preprocess
import os
import shutil


class Executor:

    def __init__(self, cf):
        unet = Modified3DUNet(1, 1)
        resnet = resnet34()
#        device = torch.device('cpu')
        device = os.environ['CUDA_VISIBLE_DEVICES']

        try:
            device = int(device)
            device = 0
            unet.load_state_dict(torch.load(cf.pathToSegmentator))
            resnet.load_state_dict(torch.load(cf.pathToClassifier))
        except ValueError:
            print("Using cpu")
            device = "cpu" 
            unet.load_state_dict(torch.load(cf.pathToSegmentator, map_location="cpu"))
            resnet.load_state_dict(torch.load(cf.pathToClassifier, map_location="cpu"))

        unet.eval()
        resnet.eval()


        self.segmentator = Estimator(unet, save_folder='./experiments/unet_full_pipe_eval/',
                                     cuda_device=device,
                                     optimizer=Adam, loss_fn=dice_loss)
        self.classify = Estimator(resnet, save_folder='./experiments/res_full_pipe_eval/',
                                  cuda_device=device,
                                  optimizer=Adam, loss_fn=torch.nn.CrossEntropyLoss())

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

    def unpack(self, pathToArchive, pathToConverted, numWorkers=3):
        prep = Preprocess(pathToArchive, pathToConverted, numWorkers)
        prep.start()
        self.pipe.add_dataset(pathToConverted)

    def start(self):
        return self.pipe.start_inference()


if __name__ == '__main__':
    conf = Config('/mnt/results/experiments')
    e = Executor(cf=conf)
    e.unpack('/mnt/archives/out.zip', '/mnt/archives/tmp')
    e.start()
