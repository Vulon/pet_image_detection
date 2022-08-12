import torch
from torch import nn
from transformers import MobileViTFeatureExtractor, MobileViTForSemanticSegmentation


class DeeplabModel(nn.Module):
    def __init__(self, pretrained_name: str, loss_function: nn.Module, target_categories: list[int]) -> None:
        super().__init__()
        # feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-small")
        self.model = MobileViTForSemanticSegmentation.from_pretrained(pretrained_name)
        self.loss_function = loss_function
        self.target_categories = target_categories

        
    def forward(self, image: torch.Tensor, labels: torch.Tensor = None):
        # shape: batch X categories X height 32 X width 32
        output = self.model(image).logits
        output = output[:, self.target_categories, :, :]
        output = torch.clamp(output, min=0, max=1)
        if labels is not None:
            loss = self.loss_function(output.float(), labels.float())
            return (loss, output)
        else:
            return (output,)


def get_model_class(pretrained_name: str):
    table = {
        "apple/deeplabv3-mobilevit-small": DeeplabModel
    }
    model_class = table.get(pretrained_name, None)
    return model_class

def get_feature_extractor(pretrained_name: str):
    table = {
        "apple/deeplabv3-mobilevit-small": MobileViTFeatureExtractor
    }
    extractor_class = table.get(pretrained_name, None)
    return extractor_class.from_pretrained(pretrained_name)