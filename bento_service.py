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
import soundfile as sf
import io

from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput, JsonOutput
from bentoml.service.artifacts.common import PickleArtifact

from espnet2.tasks.asr import ASRTask
from bento_deploy.asr_inference import Speech2Text
from espnet2.torch_utils.device_funcs import to_device
# from espnet2.bin.asr_align import CTCSegmentation, CTCSegmentationResult

#TODO: commit changes in asr_inference

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@bentoml.env(infer_pip_packages=True, setup_sh="""
#!/bin/bash
set -e

apt-get -qq install libsndfile1-dev
""")
@bentoml.artifacts([PytorchModelArtifact('classifier'), PickleArtifact('asr_train_args')])
class PyTorchFashionClassifier(bentoml.BentoService):
    
    @bentoml.utils.cached_property  # reuse transformer
    def speech2text(self):
        from bento_deploy.asr_inference import Speech2Text
        return Speech2Text('', asr_model = self.artifacts.classifier, asr_train_args= self.artifacts.asr_train_args, nbest=5, device="cpu", ctc_weight=1.0)

    def save_audio_to_file(self, fs):
        os.makedirs("./temp_audio", exist_ok=True)
        fname = os.path.join("./temp_audio", ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".wav")
        with open(fname, 'wb') as file:
            file.write(fs.read())
        print(fname)
        # file.save(fname)
        if os.path.isfile(fname):
            return fname


    @bentoml.api(input=FileInput(), output=JsonOutput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        from bento_deploy.asr_inference import Speech2Text
        
        img_tensors = []
        for fs in file_streams:
            # fname = self.save_audio_to_file(fs)
            tmp = io.BytesIO(fs.read())
            print(tmp)
            data, samplerate = sf.read(tmp)
            print(data)
            data = librosa.resample(data, samplerate, 16000) # make sure fs is 16khz

            output = self.speech2text(data)
            candidates = []
            for o in output:
                candidates.append({"hypothesis": o[0], "score": np.exp(o[-1].score.item())})
            json = jsonify(candidates)
            # img = Image.open(fs).convert(mode="L").resize((28, 28))
            # img_tensors.append(self.transform(img))
            # outputs = self.artifacts.classifier(torch.stack(img_tensors))
            # _, output_classes = outputs.max(dim=1)
        print(candidates)
        return candidates


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
    # copy("conf.json", saved_path + '/MachineTranslationService')
    
