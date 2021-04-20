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
        gc.model = Speech2Text(gc.config, gc.model_path, nbest=5, device="cuda")
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
        segments = aligner(speech=speech, text=text, fs=fs, name="dummy_name")

        # Write to "segments" file or stdout
        segments.print_utterance_text = True
        segments.print_confidence_score = True
        segments_str = str(segments)
        print(segments_str)
        # output.write(segments_str)
        return jsonify(segments_str)


        ########################################################################## 

        file = request.files['audio_data']
        fname = save_audio_to_file(file)
        data, fs = librosa.load(fname, 16000) # make sure fs is 16khz
        asr = get_inference_model()
        with torch.no_grad():

            # prepare batch
            speech = data
            # Input as audio signal
            if isinstance(speech, np.ndarray):
                speech = torch.tensor(speech)
            print("SPEECH SHAPE", speech.shape)

            # data: (Nsamples,) -> (1, Nsamples)
            speech = speech.unsqueeze(0).to(getattr(torch, asr.dtype))
            # lenghts: (1,)
            speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            batch = {"speech": speech, "speech_lengths": speech_lengths}

            # a. To device
            batch = to_device(batch, device=asr.device)
            print(asr.device)

            # b. Forward Encoder
            enc, _ = asr.asr_model.encode(**batch)
            assert len(enc) == 1, len(enc)

            # # c. Passed the encoder result and the beam search
            # nbest_hyps = self.beam_search(
            #     x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            # )
            # nbest_hyps = nbest_hyps[: self.nbest]

                # # Encode input frames
                # enc_output = asr.asr_model.encode(torch.as_tensor(feat).to(device)).unsqueeze(0)
            # Apply ctc layer to obtain log character probabilities
            lpz = asr.asr_model.ctc.log_softmax(enc)[0].cpu().numpy()

            print(lpz.shape)
            print(lpz)
            # Prepare the text for aligning
            # ground_truth_mat, utt_begin_indices = prepare_text(config, text[name])
            # # Align using CTC segmentation
            # timings, char_probs, state_list = ctc_segmentation(
            #     config, lpz, ground_truth_mat
            # )
            # logging.debug(f"state_list = {state_list}")
            # # Obtain list of utterances with time intervals and confidence score
            # segments = determine_utterance_segments(
            #     config, utt_begin_indices, char_probs, timings, text[name]
            # )
             # c. Passed the encoder result and the beam search
            nbest_hyps = asr.beam_search(
                x=enc[0], maxlenratio=asr.maxlenratio, minlenratio=asr.minlenratio
            )
            nbest_hyps = nbest_hyps[: asr.nbest]

            results = []
            for hyp in nbest_hyps:
                # assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                token_int = hyp.yseq[1:-1].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0, token_int))

                # Change integer-ids to tokens
                token = asr.converter.ids2tokens(token_int)

                if asr.tokenizer is not None:
                    text = asr.tokenizer.tokens2text(token)
                else:
                    text = None
                results.append((text, token, token_int, hyp))
            # print(results[0][1])
            # # print(len(results[0]))
            # # hyp = results[0][3]
            # # print(results[0][3].states)
            # print(hyp.states['ctc'])
            # print(hyp.states['ctc'][0].shape)
            # print(hyp.states['decoder'][0].shape)
            # print(enc.shape)
            # assert check_return_type(results)


            # apply configuration
            config = CtcSegmentationParameters()
            config.subsampling_factor = 6
            # config.replace_spaces_with_blanks = True
            config.frame_duration_ms = 8 
            # if args.subsampling_factor is not None:
            #     config.subsampling_factor = args.subsampling_factor
            # if args.frame_duration is not None:
            #     config.frame_duration_ms = args.frame_duration
            # if args.min_window_size is not None:
            #     config.min_window_size = args.min_window_size
            # if args.max_window_size is not None:
            #     config.max_window_size = args.max_window_size
            config.char_list = asr.asr_model.token_list
            config.tokenized_meta_symbol = "_"
            # if args.use_dict_blank is not None:
            #     logging.warning(
            #         "The option --use-dict-blank is deprecated. If needed,"
            #         " use --set-blank instead."
            #     )
            # if args.set_blank is not None:
            #     config.blank = args.set_blank
            # if args.replace_spaces_with_blanks is not None:
            #     if args.replace_spaces_with_blanks:
            #         config.replace_spaces_with_blanks = True
            #     else:
            #         config.replace_spaces_with_blanks = False
            # if args.gratis_blank:
            #     config.blank_transition_cost_zero = True
            # if config.blank_transition_cost_zero and args.replace_spaces_with_blanks:
            #     logging.error(
            #         "Blanks are inserted between words, and also the transition cost of blank"
            #         " is zero. This configuration may lead to misalignments!"
            #     )
            # if args.scoring_length is not None:
            #     config.score_min_mean_over_L = args.scoring_length
            print(config)
            # Prepare the text for aligning
            hyp_text = hyp[1]
            text = request.form['text']

            text_list = text.split()
            utt_tokens_list = []
            for utt in text_list:
                tokens = " ".join(asr.tokenizer.text2tokens(utt))
                print(tokens)
                token_ids = [0] + asr.converter.tokens2ids(tokens) + [0]
                utt_tokens_list.append(tokens)

            # tokens = asr.tokenizer.text2tokens(text)
            # token_ids = [-1, 0] + asr.converter.tokens2ids(tokens) + [0]
            # token_ids = np.array(token_ids).reshape(-1, 1)
            # print(token_ids)
            # print(tokens)
            # print("!!!!!!", results[0][1])

            # ground_truth_mat, utt_begin_indices = prepare_text(config, [tokens], char_list=asr.asr_model.token_list)
            # # Align using CTC segmentation
            # print(ground_truth_mat)
            # print(results[0][2])
            # print(lpz)
            # print(config)
            # ground_truth = np.array(results[0][2])
            # print(len(ground_truth))
            
            def prepare_tokenized_text(config, text, char_list=None):
                """Prepare the given tokenized text for CTC segmentation.
                :param config: an instance of CtcSegmentationParameters
                :param text: string with tokens separated by spaces
                :param char_list: a set or list that includes all characters/symbols,
                                    characters not included in this list are ignored
                :return: label matrix, character index matrix
                """
                # temporary compatibility fix for previous espnet versions
                if type(config.blank) == str:
                    config.blank = 0
                if char_list is not None:
                    config.char_list = char_list
                blank = config.char_list[config.blank]
                ground_truth = [config.start_of_ground_truth]
                utt_begin_indices = []
                for utt in text:
                    # One space in-between
                    if not ground_truth[-1] == config.space:
                        ground_truth += [config.space]
                    # Start new utterance remember index
                    utt_begin_indices.append(len(ground_truth) - 1)
                    # Add tokens of utterance
                    for token in utt.split():
                        if token in config.char_list:
                            if config.replace_spaces_with_blanks and not token.startswith(
                                config.tokenized_meta_symbol
                            ):
                                ground_truth += [config.space]
                            ground_truth += [token]
                # Add space to the end
                if not ground_truth[-1] == config.space:
                    ground_truth += [config.space]
                logging.debug(f"ground_truth: {ground_truth}")
                utt_begin_indices.append(len(ground_truth) - 1)
                # Create matrix: time frame x number of letters the character symbol spans
                max_char_len = 1
                ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
                for i in range(1, len(ground_truth)):
                    if ground_truth[i] == config.space:
                        ground_truth_mat[i, 0] = config.blank
                    else:
                        char_index = config.char_list.index(ground_truth[i])
                        ground_truth_mat[i, 0] = char_index
                return ground_truth_mat, utt_begin_indices
            
            # def prepare_tokenized_text(config, text, char_list=None):
            #     """Prepare the given tokenized text for CTC segmentation.
            #     :param config: an instance of CtcSegmentationParameters
            #     :param text: string with tokens separated by spaces
            #     :param char_list: a set or list that includes all characters/symbols,
            #                         characters not included in this list are ignored
            #     :return: label matrix, character index matrix
            #     """
            #     # temporary compatibility fix for previous espnet versions
            #     print("prep", text)
            #     if type(config.blank) == str:
            #         config.blank = 0
            #     if char_list is not None:
            #         config.char_list = char_list
            #     blank = config.char_list[config.blank]
            #     ground_truth = [config.start_of_ground_truth]
            #     ground_truth = []
            #     utt_begin_indices = []
            #     for utt in text:
            #         # One space in-between
            #         # if not ground_truth[-1] == config.space:
            #         #     ground_truth += [config.space]
            #         # Start new utterance remember index
            #         utt_begin_indices.append(len(ground_truth))
            #         # Add tokens of utterance
            #         for token in utt.split():
            #             if token in config.char_list:
            #                 if config.replace_spaces_with_blanks and not token.startswith(
            #                     config.tokenized_meta_symbol
            #                 ):
            #                     ground_truth += [config.space]
            #                 ground_truth += [token]
            #                 print(token)
            #     # Add space to the end
            #     # if not ground_truth[-1] == config.space:
            #     #     ground_truth += [config.space]
            #     logging.debug(f"ground_truth: {ground_truth}")
            #     print(ground_truth)
            #     utt_begin_indices.append(len(ground_truth))
            #     # Create matrix: time frame x number of letters the character symbol spans
            #     max_char_len = 1
            #     ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
            #     for i in range(0, len(ground_truth)):
            #         if ground_truth[i] == config.space:
            #             ground_truth_mat[i, 0] = config.blank
            #         else:
            #             char_index = config.char_list.index(ground_truth[i])
            #             ground_truth_mat[i, 0] = char_index
            #     return ground_truth_mat, utt_begin_indices

            tokens_string = " ".join(tokens)
            tokens = tokens
            print("TOKENS", tokens)
            ground_truth_mat, utt_begin_indices = prepare_tokenized_text(config, utt_tokens_list, char_list=asr.asr_model.token_list)
            print(ground_truth_mat)
            print("BG", utt_begin_indices)
            print(np.argmax(lpz, axis=1))
            spikes = np.argmax(lpz, axis=1)
            print(np.argmax(lpz, axis=1).shape)

            # np.set_printoptions(threshold=np. inf)
            print(lpz)


            timings, char_probs, state_list = ctc_segmentation(
                config, lpz, ground_truth_mat
            )
            print("TIMINGS", timings)
            print(timings, char_probs, state_list)
            utt_begin_indices = np.array(utt_begin_indices)
            print(utt_begin_indices)
            segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, utt_tokens_list
            )
            print("SEGMENTS", segments)
            print(list(zip(text_list, segments)))
            print("\n\n\n\nTIMINGS CTC", list(zip(timings, state_list)))
            print(len(state_list))
            print(len(timings))
            print(len(char_probs))

            print(np.argmax(lpz, axis=1))
            spikes = np.argmax(lpz, axis=1)
            print(np.argmax(lpz, axis=1).shape)
            # logging.debug(f"state_list = {state_list}")

            segments = []
            is_segment = False
            for idx in range(lpz.shape[0]):
                if spikes[idx] != 0:
                    segments.append({
                                     "token": asr.asr_model.token_list[spikes[idx]],
                                     "timing": idx + 1
                                     })

            # print(segments)
            plt.plot(lpz)
            plt.savefig("foo1.png")
            plt.figure()

            S = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=512, hop_length=128,
                                    fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs, hop_length=128,
                         fmax=8000, ax=ax)
            print(S.shape)

            for segment in segments:
                segment["timing_s"] = segment["timing"] * 128 / fs * (6)
                print("timi", segment["timing"] * 128 / fs * (6) )
                ax.axvline(x=(segment["timing"] ) * 128 / fs * (6) )
            print(segments)
            
            plt.savefig("foo2.png")

            def get_response_image(image_path):
                pil_img = Image.open(image_path, mode='r') # reads the PIL image
                byte_arr = io.BytesIO()
                pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
                encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
                return encoded_img
        
            # server side code
            image_path = "foo2.png" # point to your image location
            encoded_img = get_response_image(image_path)
            my_message = 'here is my message' # create your message as per your need
            response =  { 'Status' : 'Success', 'timings': segments , 'ImageBytes': encoded_img}
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
    loss = res[1]["loss_att"].item()
    response = {
        "text": text,
        "loss": loss
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
    
