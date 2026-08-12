#Prediction
import torch

from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose


def predict_binary(model, image, device, roi_size, sw_batch_size,
                   prediction_sigmoid, prediction_threshold):
    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
        AsDiscrete(threshold=prediction_threshold),
    ])

    model.eval()

    with torch.no_grad():

        output = sliding_window_inference(
            image.to(device),
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=model,
        )

        prediction = post_pred(output)

    return prediction
