from typing import BinaryIO, List

import bentoml
from PIL import Image
import torch
import os
import string
import random
import librosa
from flask import jsonify
import numpy as np

from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput, JsonOutput
from bentoml.service.artifacts.common import PickleArtifact

from espnet2.tasks.asr import ASRTask
from espnet2.bin.asr_inference import Speech2Text
from espnet2.torch_utils.device_funcs import to_device
from bento_service import PyTorchFashionClassifier

if __name__ == '__main__':
    print('hello')
    # load the model
    cfg = "/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/config.yaml"
    model_path="/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/valid.acc.best.pth"

    dtype = "float32"
    device = 'cpu'
    asr_model, asr_train_args = ASRTask.build_model_from_file(cfg, model_path, device)
    asr_model.to(dtype=getattr(torch, dtype)).eval()
    print(type(asr_train_args))
    pass

    # classifier = Speech2Text(cfg, model_path, nbest=5, device="cpu", ctc_weight=1.0)
    # pack the model
    bento_svc = PyTorchFashionClassifier()
    bento_svc.pack('classifier', asr_model)
    bento_svc.pack('asr_train_args', asr_train_args)

    # 3) save your BentoSerivce
    saved_path = bento_svc.save()

    # save bento service
