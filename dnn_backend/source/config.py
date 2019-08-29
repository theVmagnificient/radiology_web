


class Config:
    def __init__(self, s_path):
        self.pathToClassifier = '../weights/resnet_luna+dicom/last.pth.tar'
        self.pathToSegmentator = '../weights/unet_luna+dicom/last.pth.tar'

        self.spacing = [1., 1., 1.]

        self.shape = (500, 300, 300)

        self.device_id = 0

        self.crop_diam = 4

        self.crop_size = (64, 64, 64)
        self.strides = (55, 55, 55)

        self.root_path_to_save = s_path

        self.save_path = None


