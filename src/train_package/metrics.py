import numpy as np
import torch
from torch import nn


def create_compute_metrics_function():
    def compute_metrics(evalPrediction):
        # batch X categories X height X width
        with torch.no_grad():
            prediction = evalPrediction.predictions
            y = evalPrediction.label_ids

            # true_mask = torch.round(y).to(torch.bool)
            # predicted_mask = torch.round(prediction).to(torch.bool)
            # jaccard = torch.bitwise_and(predicted_mask, true_mask).sum() / torch.bitwise_or(predicted_mask, true_mask).sum()
            del evalPrediction
            true_mask = np.round(y).astype(np.bool)
            predicted_mask = np.round(prediction).astype(np.bool)
            jaccard = (
                np.bitwise_and(predicted_mask, true_mask).sum()
                / np.bitwise_or(predicted_mask, true_mask).sum()
            )
            true_mean = true_mask.mean().item()
            predicted_mean = predicted_mask.mean().item()
            del predicted_mask, true_mask
            nominator = 2 * (y * prediction).mean() + 1e-6
            denominator = (y**2).mean() + (prediction**2).mean() + 1e-6
            dice_loss = 1 - nominator / denominator
            del y, prediction
            return {
                "Jaccard": jaccard,
                "dice": np.clip(dice_loss, 0, 1),
                "true_mean": true_mean,
                "predicted_mean": predicted_mean,
            }

    return compute_metrics
