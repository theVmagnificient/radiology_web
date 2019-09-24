""" Predicting probability by xgboost."""

import torch
import logging
from utils import load_pkl


class XGBoostPredictor:

    def __init__(self, segmentator, pca_model, xgb_model):
        """ Make XGBoost predictin of probability.

        Parameters:
        -----------
        segmentator : torch.Module
            segmentation model (3D unet) to fetch bottleneck
        pca_model : sklearn.decomposition.pca.PCA
            pca model for feature reduction
        xgb_model : xgboost.sklearn.XGBRegressor
            xgboost model for predicting probabilities
            on pca-transformed features

        """
        self.segmentator = segmentator.eval()
        self.pca_model = pca_model
        self.xgb_model = xgb_model

    def xgboost_predictor(self, inputs):
        """ Make prediction of probability from inputs."""
        bttnk_features = self.segmentator.encoders(inputs)
        bttnk_features = bttnk_features.reshape(bttnk_features.shape[0], -1)
        transformed_features = self.pca_model.transform(bttnk_features)
        prediction = self.xgb_model.predict(transformed_features)
        return prediction


if __name__ == "__main__":
    logger = logging.getLogger(__name__ + '.' + str('XGB'))

    segmentator = torch.load('../../../weights/segmentation/unet3d/weights.tar')
    pca_model = load_pkl('../../../weights/classification/unet3d_xgb/pca.pkl')
    xgb_model = load_pkl('../../../weights/classification/unet3d_xgb/xgb.pkl')
    clf_xgb = XGBoostPredictor(segmentator, pca_model, xgb_model)

    # test
    inp = torch.randn(1, 1, 32, 64, 64).cuda()
    output = clf_xgb.xgboost_predictor(inp)
    logger.info(str(output))
