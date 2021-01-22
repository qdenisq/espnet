# RESULTS

- [RESULTS](#results)
  - [Model: Conformer + Librispeech](#model-conformer--librispeech)
    - [Librispeech](#librispeech)
      - [WER](#wer)
      - [CER](#cer)
      - [TER](#ter)
    - [Myst](#myst)
      - [WER](#wer-1)
      - [CER](#cer-1)
      - [TER](#ter-1)
  - [Model: Conformer + Librispeech + Myst (Finetuning)](#model-conformer--librispeech--myst-finetuning)
    - [Librispeech](#librispeech-1)
      - [WER](#wer-2)
      - [CER](#cer-2)
      - [TER](#ter-2)
    - [Myst](#myst-1)
      - [WER](#wer-3)
      - [CER](#cer-3)
      - [TER](#ter-3)

## Model: Conformer + Librispeech

- exp: `Librispeech / asr_train_asr_conformer_1_raw_bpe5000_sp`
- date: `Tue Jan  5 21:23:57 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.4`
- pytorch version: `pytorch 1.7.0`
- Git hash: `dc3f2eb2a5d77a5e389e2a66445bc66d9dc8fee2`
  - Commit date: `Sat Dec 12 20:20:49 2020 +0900`
- path: `espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.best.pth`


### Librispeech

#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|54402|97.2|2.5|0.3|0.4|**3.2**|35.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|50948|93.1|6.3|0.6|0.8|**7.8**|54.8|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|52576|97.1|2.6|0.3|0.4|**3.4**|36.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|52343|93.2|6.2|0.7|1.0|**7.8**|56.1|

#### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|288456|99.2|0.5|0.4|0.4|1.2|35.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|265951|97.4|1.5|1.0|0.9|3.5|54.8|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|281530|99.2|0.4|0.4|0.4|1.2|36.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|272758|97.6|1.4|1.0|1.0|3.4|56.1|
</details>

#### TER
<details>
<summary>Click to expand!</summary>
<p>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/dev_clean|2703|68010|96.7|2.4|0.9|0.6|3.9|35.3|
|decode_asr_asr_model_valid.acc.best/dev_other|2864|63110|91.8|6.1|2.2|1.2|9.4|54.8|
|decode_asr_asr_model_valid.acc.best/test_clean|2620|65818|96.6|2.4|1.0|0.6|4.0|36.0|
|decode_asr_asr_model_valid.acc.best/test_other|2939|65101|91.9|5.8|2.3|1.1|9.2|56.1|
</p>
</details>


### Myst
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|5113|63499|74.8|19.7|5.5|5.7|**30.9**|87.8|

#### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|5113|320254|87.7|5.8|6.5|5.7|18.0|87.8|
</details>

#### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|5113|113078|22.0|47.0|31.0|2.0|80.0|96.6|
</details>

## Model: Conformer + Librispeech + Myst (Finetuning)

- exp: `Myst / asr_train_asr_conformer_1_raw_bpe5000_sp`
- date: `Sat Jan  9 20:47:27 AWST 2021`
- python version: `3.6.10 (default, Dec 19 2019, 23:04:32)  [GCC 5.4.0 20160609]`
- espnet version: `espnet 0.9.4`
- pytorch version: `pytorch 1.7.0`
- Git hash: `dc3f2eb2a5d77a5e389e2a66445bc66d9dc8fee2`
  - Commit date: `Sat Dec 12 20:20:49 2020 +0900`
- path: `espnet/egs2/myst/asr1/exp/asr_train_asr_conformer_1_raw_bpe5000_sp/valid.acc.ave_10best.pth`


### Librispeech

#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|54402|69.6|29.3|1.1|3.8|**34.1**|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|50948|49.8|48.1|2.1|6.8|**57.0**|98.0|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|52576|69.2|29.6|1.2|4.0|**34.8**|95.3|
|decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|52343|49.9|47.9|2.1|6.7|**56.8**|98.1|

#### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|288456|83.0|8.7|8.4|3.3|20.4|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|265951|67.2|14.0|18.8|4.3|37.1|98.0|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|281530|82.9|8.9|8.2|3.4|20.5|95.3|
|decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|272758|67.1|14.2|18.7|4.2|37.1|98.1|

</details>


#### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev_clean|2703|68010|67.0|27.5|5.6|7.5|40.5|95.7|
|decode_asr_asr_model_valid.acc.ave_10best/dev_other|2864|63110|47.2|46.8|6.0|20.8|73.6|98.0|
|decode_asr_asr_model_valid.acc.ave_10best/test_clean|2620|65818|67.1|27.6|5.2|7.6|40.5|95.3|
|decode_asr_asr_model_valid.acc.ave_10best/test_other|2939|65101|46.7|47.2|6.1|19.6|72.9|98.1|

</details>

### Myst
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|5013|66376|92.0|5.6|2.4|2.3|**10.3**|54.9|
|decode_asr_asr_model_valid.acc.ave_10best/test|5113|63499|90.6|6.6|2.9|3.2|**12.7**|55.3|

#### CER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|5013|333632|96.2|1.3|2.4|2.3|6.0|54.9|
|decode_asr_asr_model_valid.acc.ave_10best/test|5113|320254|95.3|1.7|3.0|3.2|7.9|55.3|
</details>

#### TER
<details>
<summary>Click to expand!</summary>

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|5013|87933|92.5|4.2|3.3|2.7|10.2|54.9|
|decode_asr_asr_model_valid.acc.ave_10best/test|5113|83972|91.1|5.1|3.8|3.7|12.6|55.3|
</details>