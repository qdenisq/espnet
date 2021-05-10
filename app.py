from flask import Flask, render_template, g, request, redirect, url_for, jsonify, Response
from datetime import datetime
import requests
import os
import soundfile
import librosa
import librosa.display
import torch
import numpy as np
from collections import defaultdict
import random
import string
import argparse
from matplotlib import pyplot as plt
import logging
import io
from base64 import encodebytes
from PIL import Image

from espnet2.bin.asr_inference import Speech2Text
from espnet2.torch_utils.device_funcs import to_device
from espnet2.bin.asr_align import CTCSegmentation, CTCSegmentationResult

# imports for CTC segmentation
from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text

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
        #TODO: implement consistent mixed scoring (att+CTC) for all requests (i.e. recognize, validate, align)
        gc.model = Speech2Text(gc.config, gc.model_path, nbest=5, device="cuda", ctc_weight=1.0)
    return gc.model

def get_aligner():
    if not hasattr(gc, "aligner"):
        model_config = {"asr_train_config": gc.config,
                        "asr_model_file": gc.model_path}
        # dummy config for now
        kwargs = {
            'kaldi_style_text': False
        }
        aligner = CTCSegmentation(**model_config, **kwargs)
        gc.aligner = aligner
    return gc.aligner

def save_audio_to_file(file):
    os.makedirs("./temp_audio", exist_ok=True)
    fname = os.path.join("./temp_audio", ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".wav")
    file.save(fname)
    if os.path.isfile(fname):
        app.logger.debug(f"audio file {fname} created")
    return fname

@app.after_request
def after_request(response):
    allowed_origins = ["http://192.168.21.235:5002", "http://192.168.21.235:5003"]
    response.headers.add("Access-Control-Allow-Origin", allowed_origins)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['POST'])
def recognise():
    file = request.files['audio_data']
    fname = save_audio_to_file(file)
    data, fs = librosa.load(fname, 16000) # make sure fs is 16khz
    asr = get_inference_model()
    output = asr(data)
    candidates = []
    for o in output:
        candidates.append({"hypothesis": o[0], "score": np.exp(o[-1].score.item())})
    json = jsonify(candidates)
    app.logger.debug(f"recognition response: {candidates}")
    return json

@app.route('/align', methods=['POST'])
def align():
    if request.method == 'POST':
        aligner = get_aligner()
        # load audio file
        file = request.files['audio_data']
        fname = save_audio_to_file(file)
        speech, fs = librosa.load(fname, 16000) # make sure fs is 16khz

        text = request.form['text']
        # perform inference and CTC segmentation
        ctc_segmentation_result = aligner(speech=speech, text=text, fs=fs, name="dummy_name")
        response = []
        final_score = 1.
        for i, segment in enumerate(ctc_segmentation_result.segments):
            segment_score = segment[2]
            final_score *= segment_score
            segment_text = ctc_segmentation_result.text[i]
            segment_start = segment[0]
            segment_end = segment[1]
            response.append({
                "score": segment_score,
                "text": segment_text,
                "start": segment_start,
                "end": segment_end
            })
        print(response)
        print(final_score)
        return jsonify(response)

@app.route('/validate', methods=['POST'])
def validate():
    file = request.files['audio_data']
    fname = save_audio_to_file(file)
    data, fs = librosa.load(fname, 16000) # make sure fs is 16khz
    asr = get_inference_model()

    if 'text' not in request.form:
        return Response(f"Missing 'text' field", 400)

    # loss calc
    text = request.form['text']
    # prepare batch
    speech = data
    # Input as audio signal
    if isinstance(speech, np.ndarray):
        speech = torch.tensor(speech)
    # data: (Nsamples,) -> (1, Nsamples)
    speech = speech.unsqueeze(0).to(getattr(torch, asr.dtype))
    # lenghts: (1,)
    speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    tokens = asr.tokenizer.text2tokens(text)
    # Change integer-ids to tokens
    token_ids = asr.converter.tokens2ids(tokens)
    text_tensor = torch.tensor(token_ids)
    text_tensor = text_tensor.unsqueeze(0)
    text_lengths = torch.tensor(text_tensor.shape[-1]).unsqueeze(0)
    batch = {"speech": speech,
                "speech_lengths": speech_lengths,
                "text": text_tensor,
                "text_lengths": text_lengths}

    batch = to_device(batch, device=asr.device)

    res = asr.asr_model(**batch)
    loss = res[1]["loss_ctc"].item()
    score = np.exp(-loss)
    response = {
        "text": text,
        "score": score
        }
    json = jsonify(response)
    app.logger.debug(f"validate response: {response}")
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
    
    # Librispeech
    # cfg="/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/config.yaml"
    # model_path = "/home/denis/bin/projects/ESPnet/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.best.pth"

    # Librimyst_0
    cfg = "/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_MYST_Pretrained_0/config.yaml"
    model_path="/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_MYST_Pretrained_0/valid.acc.best.pth"

    # Librimyst_1
    cfg = "/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/config.yaml"
    model_path="/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/valid.acc.best.pth"
    
    # Myst
    # cfg="/home/denis/bin/projects/ESPnet/espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/config.yaml"
    # model_path = "/home/denis/bin/projects/ESPnet/espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.ave_10best.pth"

    gc.config=cfg
    gc.model_path=model_path

    app.run(host="192.168.21.235", debug=True, port=5000)
    
