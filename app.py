from flask import Flask, render_template, g, request, redirect, url_for, jsonify
from datetime import datetime
import requests
import os
import soundfile
# import yaml
import librosa
import torch
# from torchvision.transforms import Compose
import numpy as np
from collections import defaultdict
import random
import string
import argparse

from espnet2.bin.asr_inference import Speech2Text

# from src.triplet_net.model import TripletDeepLSTMNet
# from src.triplet_net.loss import TripletNetLoss
# from src.datasets.LibriSpeechWords.dataset import LibriSpeechWordsTripletDataset, pad_collate_triplet
# from src.transform.audio_processing import AudioPreprocessorMFCCDeltaDelta
# from src.transform.common import *
# from src.triplet_net.train import get_embeddings, str_to_class

# from src.inference_model import InferenceModel, TripletNetInferenceModel, DeepSpeechInferenceModel


app = Flask(__name__)

class GlobalContext():
    def __init__(self):
        pass
gc = GlobalContext()


def get_config():
    if not hasattr(gc, "config"):
        raise ValueError('config is not defined')
    return gc.config

def get_inference_model():
    if not hasattr(gc, "model"):
        gc.model = Speech2Text(gc.config, gc.model_path)
    return gc.model

# def get_words():
#     if not hasattr(gc, "words"):
#         gc.words = gc.config['words']
#     return gc.words


@app.after_request
def after_request(response):
    allowed_origins = ["http://192.168.21.235:5002", "http://192.168.21.235:5003"]
    response.headers.add("Access-Control-Allow-Origin", allowed_origins)
    # response.headers.add("Access-Control-Allow-Origin", "http://192.168.21.235:5003")

    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# @app.route('/words', methods=['GET'])
# def words():
#     words = get_words()
#     return jsonify(words)

@app.route('/', methods=['GET', 'POST'])
def score():
    if request.method == 'POST':

        file = request.files['audio_data']
        os.makedirs("./temp_audio", exist_ok=True)
        fname = os.path.join("./temp_audio", ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".wav")
        file.save(fname)
        if os.path.isfile(fname):
            print(f"{fname} exists")

        data, fs = librosa.load(fname, 16000) # make sure fs is 16khz
        # data = librosa.util.normalize(data)
        print(fs)
        
        output = get_inference_model()(data)

        # os.remove(fname)
        print(output[0][0])
        json = jsonify(output[0][0])
        print(json)
        return json

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-config', '-C', help="path to the config file")
    # parser.add_argument('-model', '-M', help="path to the saved model")
    # args = parser.parse_args()

    # config_path = args.config
    # model_path = args.model

    #TODO: remove hardcoded cfg and model_path

    # gc.config = config_path
    # gc.model_path = model_path

    # cfg="/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp/config.yaml"
    # model_path = "/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp/valid.acc.ave_10best.pth"
# egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp
    
    cfg="/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/config.yaml"
    model_path = "/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.best.pth"

    # cfg="/home/denis/bin/projects/ESPnet/espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/config.yaml"
    # model_path = "/home/denis/bin/projects/ESPnet/espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.ave_10best.pth"



    gc.config=cfg
    gc.model_path=model_path

    app.run(host="192.168.21.235", debug=True, port=5000)
    
