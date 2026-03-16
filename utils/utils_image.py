import os

import cv2
import numpy as np
import torch


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def imread_uint(path, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def uint2tensor4(img, data_range=1.0):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor * float(data_range)


def tensor2uint(img, data_range=1.0):
    if isinstance(img, list):
        return [tensor2uint(v, data_range=data_range) for v in img]

    if isinstance(img, torch.Tensor):
        img = img.detach().float().cpu().clamp_(0, float(data_range))
        img = img.squeeze(0).numpy()
    img = img / float(data_range)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return img


def imsave(img, img_path):
    mkdir(os.path.dirname(img_path) or ".")
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)
