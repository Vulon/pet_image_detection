import torch
from torch import nn
from transformers import MobileViTFeatureExtractor, MobileViTForSemanticSegmentation


class DeeplabScoreModel(nn.Module):
    def __init__(self, pretrained_name: str, target_categories: list[int]) -> None:
        super().__init__()
        self.model = MobileViTForSemanticSegmentation.from_pretrained(pretrained_name)
        self.target_categories = target_categories

    def forward(self, image: torch.Tensor, labels: torch.Tensor = None):
        # shape: batch X categories X height 32 X width 32
        output = self.model(image).logits
        output = output[:, self.target_categories, :, :]
        output = torch.clamp(output, min=0, max=1)

        return output


def get_model_class(pretrained_name: str):
    table = {"apple/deeplabv3-mobilevit-small": DeeplabScoreModel}
    model_class = table.get(pretrained_name, None)
    return model_class


def get_feature_extractor(pretrained_name: str):
    table = {"apple/deeplabv3-mobilevit-small": MobileViTFeatureExtractor}
    extractor_class = table.get(pretrained_name, None)
    return extractor_class.from_pretrained(pretrained_name)
