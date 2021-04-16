# RESULTS

- [RESULTS](#results)
  - [Model: Conformer + Librispeech + LibriMyst (Finetuning) (BPE_50)](#model-conformer--librispeech--librimyst-finetuning-bpe_50)
    - [WER](#wer)
    - [CER](#cer)
    - [TER](#ter)
  - [Model: Conformer + Librispeech + LibriMyst (Finetuning)](#model-conformer--librispeech--librimyst-finetuning)
    - [WER](#wer-1)
    - [CER](#cer-1)
    - [TER](#ter-1)
  - [Model: Conformer + Librispeech](#model-conformer--librispeech)
    - [WER](#wer-2)
    - [CER](#cer-2)
    - [TER](#ter-2)
  - [Model: Conformer + Librispeech + Myst (Finetuning)](#model-conformer--librispeech--myst-finetuning)
    - [WER](#wer-3)
    - [CER](#cer-3)
    - [TER](#ter-3)

## Model: Conformer + Librispeech + LibriMyst (Finetuning) (BPE_50)
- exp: `librimyst/asr1/exp/asr_DS_bpe_50`
- date: `Fri Apr 16 07:32:17 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.7.0`
- Git hash: `cc67a32e86f5a66141b2f7fe4a1b5fcfa6df676a`
  - Commit date: `Fri Feb 19 08:59:50 2021 +0800`
- path: `espnet/egs2/librimyst/asr1/exp/asr_DS_bpe_50/valid.acc.best.pth`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|54402|97.3|2.5|0.2|0.3|3.0|36.1|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|52576|97.1|2.6|0.2|0.3|3.2|36.8|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/dev_other|2864|50948|92.7|6.6|0.7|0.7|**8.0**|57.1|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/test_other|2939|52343|92.8|6.5|0.7|0.9|**8.0**|59.7|
|(**Myst**) decode_asr_asr_model_valid.acc.best/dev_myst|5745|92397|91.1|6.4|2.5|2.4|**11.3**|63.3|
|(**Myst**) decode_asr_asr_model_valid.acc.best/test_myst|5776|96310|91.6|6.1|2.3|2.5|**10.9**|61.0|

### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|288456|99.3|0.4|0.3|0.3|1.0|36.1|
|decode_asr_asr_model_valid.acc.best/dev_myst|5745|461970|96.0|1.5|2.5|2.5|6.4|63.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|265951|97.5|1.5|1.0|0.8|3.3|57.1|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|281530|99.3|0.4|0.3|0.3|1.0|36.8|
|decode_asr_asr_model_valid.acc.best/test_myst|5776|482881|96.2|1.4|2.4|2.5|6.2|61.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|272758|97.6|1.4|1.0|0.8|3.2|59.7|
</details>

### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|223512|99.0|0.6|0.4|0.3|1.3|36.1|
|decode_asr_asr_model_valid.acc.best/dev_myst|5745|355043|95.3|2.1|2.5|2.5|7.2|63.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|207533|96.7|2.2|1.1|0.9|4.2|57.1|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|218267|98.9|0.7|0.4|0.3|1.4|36.8|
|decode_asr_asr_model_valid.acc.best/test_myst|5776|372949|95.6|2.0|2.4|2.5|6.9|61.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|212174|96.8|2.1|1.2|0.8|4.1|59.7|
</details>

## Model: Conformer + Librispeech + LibriMyst (Finetuning)

- exp: `librimyst/asr1/exp/asr_DS_MYST_Pretrained_0`
- date: `Thu Feb 18 23:32:50 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.7.0`
- Git hash: `6e6cffae2215e5759f4871b0738b002097ff1142`
  - Commit date: `Fri Jan 22 12:24:52 2021 +0800`
- path: `espnet/egs2/librimyst/asr1/exp/asr_DS_MYST_Pretrained_0/valid.acc.best.pth`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|54402|97.3|2.4|0.3|0.3|3.0|34.3|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|52576|97.1|2.6|0.3|0.4|3.3|34.5|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/dev_other|2864|50948|93.2|6.1|0.7|0.8|**7.6**|54.6|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/test_other|2939|52343|93.3|6.0|0.7|0.9|**7.5**|55.8|
|(**Myst**) decode_asr_asr_model_valid.acc.best/test_myst|5776|96310|91.9|5.6|2.5|2.3|**10.4**|59.4|
|(**Myst**) decode_asr_asr_model_valid.acc.best/dev_myst|5745|92397|91.3|6.0|2.7|2.3|**11.0**|61.7|


### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|288456|99.2|0.4|0.4|0.3|1.1|34.3|
|decode_asr_asr_model_valid.acc.best/dev_myst|5745|461970|95.8|1.5|2.7|2.2|6.4|61.7|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|265951|97.5|1.5|1.0|0.9|3.4|54.6|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|281530|99.2|0.4|0.4|0.4|1.2|34.5|
|decode_asr_asr_model_valid.acc.best/test_myst|5776|482881|96.2|1.3|2.6|2.3|6.1|59.4|
|decode_asr_asr_model_valid.acc.best/test_other|2939|272758|97.7|1.4|1.0|0.9|3.2|55.8|
</details>

### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|68010|96.8|2.3|0.9|0.5|3.7|34.3|
|decode_asr_asr_model_valid.acc.best/dev_myst|5745|120901|91.9|4.6|3.5|2.6|10.7|61.7|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|63110|91.9|5.9|2.2|1.1|9.2|54.6|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|65818|96.5|2.4|1.0|0.5|4.0|34.5|
|decode_asr_asr_model_valid.acc.best/test_myst|5776|126656|92.5|4.1|3.3|2.8|10.2|59.4|
|decode_asr_asr_model_valid.acc.best/test_other|2939|65101|92.0|5.6|2.4|0.9|9.0|55.8|
</details>



## Model: Conformer + Librispeech

- exp: `librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp`
- date: `Tue Jan  5 21:23:57 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.4`
- pytorch version: `pytorch 1.7.0`
- Git hash: `dc3f2eb2a5d77a5e389e2a66445bc66d9dc8fee2`
  - Commit date: `Sat Dec 12 20:20:49 2020 +0900`
- path: `espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.best.pth`


### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|54402|97.2|2.5|0.3|0.4|3.2|35.3|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|52576|97.1|2.6|0.3|0.4|3.4|36.0|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/dev_other|2864|50948|93.1|6.3|0.6|0.8|**7.8**|54.8|
|(**Librispeech**) decode_asr_asr_model_valid.acc.best/test_other|2939|52343|93.2|6.2|0.7|1.0|**7.8**|56.1|
|(**Myst**) decode_asr_asr_model_valid.acc.best/test|5113|63499|74.8|19.7|5.5|5.7|**30.9**|87.8|

### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|288456|99.2|0.5|0.4|0.4|1.2|35.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|265951|97.4|1.5|1.0|0.9|3.5|54.8|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|281530|99.2|0.4|0.4|0.4|1.2|36.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|272758|97.6|1.4|1.0|1.0|3.4|56.1|
|decode_asr_asr_model_valid.acc.best/test|5113|320254|87.7|5.8|6.5|5.7|18.0|87.8|
</details>

### TER
<details>
<summary>Click to expand!</summary>
<p>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|68010|96.7|2.4|0.9|0.6|3.9|35.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|63110|91.8|6.1|2.2|1.2|9.4|54.8|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|65818|96.6|2.4|1.0|0.6|4.0|36.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|65101|91.9|5.8|2.3|1.1|9.2|56.1|
|decode_asr_asr_model_valid.acc.best/test|5113|113078|22.0|47.0|31.0|2.0|80.0|96.6|
</p>
</details>


## Model: Conformer + Librispeech + Myst (Finetuning)

- exp: `myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp`
- date: `Sat Jan  9 20:47:27 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.4`
- pytorch version: `pytorch 1.7.0`
- Git hash: `dc3f2eb2a5d77a5e389e2a66445bc66d9dc8fee2`
  - Commit date: `Sat Dec 12 20:20:49 2020 +0900`
- path: `espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.ave_10best.pth`


### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|54402|69.6|29.3|1.1|3.8|34.1|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|52576|69.2|29.6|1.2|4.0|34.8|95.3|
|(**Librispeech**) decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|50948|49.8|48.1|2.1|6.8|**57.0**|98.0|
|(**Librispeech**) decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|52343|49.9|47.9|2.1|6.7|**56.8**|98.1|
|(**Myst**) decode_asr_asr_model_valid.acc.ave_10best/dev|5013|66376|92.0|5.6|2.4|2.3|**10.3**|54.9|
|(**Myst**) decode_asr_asr_model_valid.acc.ave_10best/test|5113|63499|90.6|6.6|2.9|3.2|**12.7**|55.3|

### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|288456|83.0|8.7|8.4|3.3|20.4|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|265951|67.2|14.0|18.8|4.3|37.1|98.0|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|281530|82.9|8.9|8.2|3.4|20.5|95.3|
|decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|272758|67.1|14.2|18.7|4.2|37.1|98.1|
|decode_asr_asr_model_valid.acc.ave_10best/dev|5013|333632|96.2|1.3|2.4|2.3|6.0|54.9|
|decode_asr_asr_model_valid.acc.ave_10best/test|5113|320254|95.3|1.7|3.0|3.2|7.9|55.3|

</details>

### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|68010|67.0|27.5|5.6|7.5|40.5|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|63110|47.2|46.8|6.0|20.8|73.6|98.0|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|65818|67.1|27.6|5.2|7.6|40.5|95.3|
|decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|65101|46.7|47.2|6.1|19.6|72.9|98.1|
|decode_asr_asr_model_valid.acc.ave_10best/dev|5013|87933|92.5|4.2|3.3|2.7|10.2|54.9|
|decode_asr_asr_model_valid.acc.ave_10best/test|5113|83972|91.1|5.1|3.8|3.7|12.6|55.3|

</details>