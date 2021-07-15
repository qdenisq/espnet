

# . ./path.sh
# . ./cmd.sh

. ./path.sh
. ./cmd.sh

asr_config=/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/config.yaml
asr_model=/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/valid.acc.best.pth

# cfg = "/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/config.yaml"
# model_path="/home/denis/bin/projects/ESPnet/espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/valid.acc.best.pth"
# prepare the text file
wav="/home/denis/bin/projects/ESPnet/myst_002030_2014-03-11_12-53-38_LS_2.3_004.wav"
text="/home/denis/bin/projects/ESPnet/ctc_align_text.txt"
cat << EOF > ${text}
utt1 THAT
utt2 CELLS
utt3 ARE
utt4 SOMETIMES
utt5 LIVE
utt6 AND
utt7 WE'RE
utt8 TALKING
utt9 ABOUT
utt10 LIVING
utt11 ORGANISMS
EOF
# obtain alignments:
python3 -m espnet2.bin.asr_align --asr_train_config ${asr_config} --asr_model_file ${asr_model} --audio ${wav} --text ${text}