import json
import logging
import os
import shutil
import sys
import tempfile
import time
import typing
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from score_models import get_feature_extractor, get_model_class
from transformers import RobertaTokenizer


def startup_server():
    logging.info("Starting up the Image segmentation server")
    with open("./params.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_class = get_model_class(config["model"]["pretrained_name"])
    model = model_class(
        config["model"]["pretrained_name"], config["model"]["target_categories"]
    )

    model.load_state_dict(torch.load("pytorch_model.bin"))
    feature_extractor = get_feature_extractor(config["model"]["pretrained_name"])
    return model, feature_extractor, config


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model, feature_extractor, config = startup_server()


async def load_image(file: UploadFile):
    data = await file.read()
    folder = tempfile.mkdtemp()
    try:
        format = os.path.basename(file.filename).split(".")[-1]
        file_path = os.path.join(folder, f"image.{format}")
        with open(file_path, "wb") as image_file:
            image_file.write(data)
        image = cv2.imread(file_path)
        return image
    finally:
        shutil.rmtree(folder)


def upscale_mask(mask: np.ndarray, target_shape):
    scaled = cv2.resize(mask, target_shape, interpolation=cv2.INTER_CUBIC)
    scaled = np.clip(scaled, 0, 1)
    return np.round(scaled)


@app.post("/predict_file")
async def upload_file_prediction(file: UploadFile):
    image = await load_image(file)

    image_tensor = feature_extractor(images=image, return_tensors="pt")["pixel_values"]
    predicted_masks = model(image_tensor)[0]
    results = dict()
    for mask_id in config["score"]["masks_names"].keys():
        numpy_mask = predicted_masks[mask_id].detach().numpy()
        mask_name = config["score"]["masks_names"][mask_id]
        results[mask_name] = numpy_mask.flatten().tolist()
        scaled_mask = upscale_mask(numpy_mask, (image.shape[0], image.shape[1]))
        results[f"{mask_name}_scaled"] = scaled_mask.flatten().tolist()

    results["mask_width"] = predicted_masks.shape[2]
    results["mask_height"] = predicted_masks.shape[1]

    return results
