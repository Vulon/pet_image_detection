import numpy as np
import torch
from torch import nn


def create_compute_metrics_function():
    def compute_metrics(evalPrediction):
        # batch X categories X height X width
        prediction = evalPrediction.predictions
        y = evalPrediction.label_ids

        # true_mask = torch.round(y).to(torch.bool)
        # predicted_mask = torch.round(prediction).to(torch.bool)
        # jaccard = torch.bitwise_and(predicted_mask, true_mask).sum() / torch.bitwise_or(predicted_mask, true_mask).sum()
        true_mask = np.round(y).astype(np.bool)
        predicted_mask = np.round(prediction).astype(np.bool)
        jaccard = (
            np.bitwise_and(predicted_mask, true_mask).sum()
            / np.bitwise_or(predicted_mask, true_mask).sum()
        )

        dice_loss = 1 - (2 * y * prediction).sum() / (y**2 + prediction**2).sum()

        return {
            "Jaccard": jaccard,
            "dice": dice_loss,
            "true_mean": true_mask.mean(),
            "predicted_mean": predicted_mask.mean(),
        }

    return compute_metrics
