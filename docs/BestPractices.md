# Best Practices

## Fundamental concepts and materials

### Configuration files

A configuration file is a YAML file that defines enabled features, model hyperparameters and controls the behavior of the binarizer, trainer and inference. Almost all settings and controls in this repository, including the practices in this guidance, are achieved through configuration files.

For more information of the configuration system and configurable attributes, see [Configuration Schemas](ConfigurationSchemas.md).

### Languages

Each language you are dealing with should have a unique tag in the configuration file. **We highly recommend using ISO 639 language codes as language tags.** For example, `zh` and `zho` stands for Chinese (`cmn` specifically for Mandarin Chinese), `ja` and `jpn` for Japanese, `en` and `eng` for English, `yue` for Cantonese (Yue). You can download a complete language code table from https://iso639-3.sil.org/code_tables/download_tables.

### Phonemes

Phonemes are the fundamental part of dictionaries and labels. There are two types of phonemes: language-specific phonemes and global phonemes.

**Language-specific phonemes:** If there are multiple languages, all language-specific phonemes will be prefixed with its language name. For example: `zh/a`, `ja/o`, `en/eh`. These are called the **full name** of the phonemes, while `a`, `o`, `eh` are called the **short name** which has definite meaning only in a specific language context. If there is only one language, the short names can be used to determine each phoneme.

**Global phonemes:** Some phonemes do not belong to any language. There are two reserved global phoneme tags: `SP` for space, and `AP` for aspiration. There can also be other user-defined tags (`EP`, `GS`, `VF`, etc.). These tags will not be prefixed with language, and are prior when identifying phoneme names.

Extra phonemes, including user-defined global phonemes and additional language-specific phonemes that are not present in the dictionaries, can be defined in a list in the configuration file (full names should be used):

```yaml
extra_phonemes: ['EP', 'ja/cl']
```

The phoneme set expands rapidly with the number of languages. There are actually many similar phonemes that can be merged. Define the merging groups in your configuration file (full names should be used):

```yaml
merged_phoneme_groups:
  - [zh/i, ja/i, en/iy]
  - [zh/s, ja/s, en/s]
  - [ja/cl, SP]  # global phonemes can also be merged
  # ... (other groups omitted for brevity)
use_lang_id: true  # whether to use language embedding; only take effects if there are cross-lingual phonemes
```

Merging phonemes does not mean that they are exactly the same for the dictionary. For those cross-lingual merged phonemes, Setting `use_lang_id` to true will still distinguish them by language IDs.

#### Phoneme naming principles

- Short names of language-specific phonemes should not conflict with global phoneme names, including reserved ones.
- `/` cannot be used because it is already used for splitting the language tag and the short name.
- `-` and `+` cannot be used because they are defined as slur tags in most singing voice synthesis editors.
- Other special characters, including but not limited to `@`, `#`, `&`, `|`, `<`, `>`, is not recommended because they may be used as special tags in the future format changes.
- ASCII characters are preferred for the best encoding compatibility, but all UTF-8 characters are acceptable.

### Dictionaries

Each language should have a corresponding dictionary. Define languages and dictionaries in your configuration file:

```yaml
dictionaries:
  zh: dictionaries/opencpop-extension.txt
  ja: dictionaries/japanese_dict_full.txt
  en: dictionaries/ds_cmudict-07b.txt
num_lang: 3  # number of languages; should be >= number of defined languages
```

Each dictionary is a *.txt* file, in which each line represents a mapping rule from one syllable to its phoneme sequence. The syllable and the phonemes are split by `tab`, and the phonemes are split by `space`:

```
<syllable>	<phoneme1> <phoneme2> ...
```

#### Syllable naming principles

- Try to use a standard writing or pronouncing system. For example, pinyin for Mandarin Chinese, romaji for Japanese and English words for English.
- `AP` and `SP` cannot be used because they are reserved tags when using DiffSinger in editors.
- `/` cannot be used because it is already used for splitting the language tag and the short name.
- `-` and `+` cannot be used because they are defined as slur tags in most singing voice synthesis editors.
- Syllable names is not recommended to start with `.` because this may have special meanings in the future editors.
- Other special characters, including but not limited to `@`, `#`, `&`, `|`, `<`, `>`, is not recommended because they may be used as special tags in the future format changes.
- ASCII characters are preferred for the best encoding compatibility, but all UTF-8 characters are acceptable.

There are some example dictionaries in the [dictionaries/](../dictionaries) folder.

### Datasets

A dataset mainly includes recordings and transcriptions, which is called a _raw dataset_. Raw datasets should be organized as the following folder structure:

- my_raw_data/
  - wavs/
    - 001.wav
    - 002.wav
    - ... (more recording files)
  - transcriptions.csv

In the example above, the _my_raw_data_ directory is the root directory of a raw dataset.

The _transcriptions.csv_ file contains all labels of the recordings. The common column of the CSV file is `name`, which represents all recording items by their filenames **without extension**. Elements of sequence attributes should be split by `space`. Other required columns may vary according to the category of the model you are training, and will be introduced in the following sections.

Each dataset should have a main language. If you have many recordings in multiple languages, it is recommended to separate them by language (you can merge their speaker IDs in the configuration). In each dataset, the main language is set as the language context, and phoneme labels in transcriptions.csv do not need a prefix (short name). It is also valid if there are phonemes from other languages, but all of them should be prefixed with their actual language (full name). Global phonemes should not be prefixed in any datasets.

You can define your datasets in the configuration file like this:

```yaml
datasets:  # define all raw datasets
  - raw_data_dir: data/spk1-zh/raw  # path to the root of a raw dataset
    speaker: speaker1  # speaker name
    spk_id: 0  # optional; use this to merge two datasets; otherwise automatically assigned
    language: zh  # language tag (main language) of this dataset
    test_prefixes:  # optional; validation samples from this dataset
      - wav1
      - wav2
  - raw_data_dir: data/spk1-en/raw
    speaker: speaker1
    spk_id: 0  # specify the same speaker ID to merge into the previous one
    language: en
    test_prefixes:
      - wav1
      - wav2
  - raw_data_dir: data/spk2/raw
    speaker: speaker2
    language: ja
    test_prefixes:
      - wav1
      - wav2
  # ... (other datasets omitted for brevity)
num_spk: 2  # number of languages; should be > maximum speaker ID
```

### DS files

DS files are JSON files with _.ds_ suffix that contains phoneme sequence, phoneme durations, music scores or curve parameters. They are mainly used to run inference on models for test and evaluation purposes, and they can be used as training data in some cases. There are some example DS files in the [samples/](../samples) folder.

The current recommended way of using a model for production purposes is to use [OpenUTAU for DiffSinger](https://github.com/xunmengshe/OpenUtau). It can export DS files as well.

### Other fundamental assets

#### Vocoders

A vocoder is a model that can reconstruct the audio waveform given the low-dimensional mel-spectrogram. The vocoder is the essential dependency if you want to train an acoustic model and hear the voice on the TensorBoard.

The [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) provides a universal pre-trained NSF-HiFiGAN vocoder that can be used for starters of this repository. To use it, download the model (~50 MB size) from its releases and unzip it into the `checkpoints/` folder.

The pre-trained vocoder can be fine-tuned on your target dataset. It is highly recommended to do so because fine-tuned vocoder can generate much better results on specific (seen) datasets while does not need much computing resources. See the [vocoder training and fine-tuning repository](https://github.com/openvpi/SingingVocoders) for detailed instructions. After you get the fine-tuned vocoder checkpoint, you can configure it by `vocoder_ckpt` key in your configuration file. The fine-tuned NSF-HiFiGAN vocoder checkpoints can be exported to ONNX format like other DiffSinger user models for further production purposes.

Another unrecommended option: train an ultra-lightweight [DDSP vocoder](https://github.com/yxlllc/pc-ddsp) first by yourself, then configure it according to the relevant [instructions](https://github.com/yxlllc/pc-ddsp/blob/master/DiffSinger.md).

#### Feature extractors or auxiliary models

RMVPE is the recommended pitch extractor of this repository, which is an NN-based algorithm and requires a pre-trained model. For more information about pitch extractors and how to configure them, see [feature extraction](#pitch-extraction).

Vocal Remover (VR) is the recommended harmonic-noise separator of this repository, which is an NN-based algorithm and requires a pre-trained model. For more information about harmonic-noise separators and how to configure them, see [feature extraction](#harmonic-noise-separation).

## Overview: training acoustic models

An acoustic model takes low-level singing information as input, including (but not limited to) phoneme sequence, phoneme durations and F0 sequence. The only output of an acoustic model is the mel-spectrogram, which can be converted to waveform (the final audio) through the vocoder. Briefly speaking, an acoustic model takes in all features that are explicitly given, and produces the singing voice.

### Datasets

To train an acoustic model, you must have three columns in your transcriptions.csv: `name`, `ph_seq` and `ph_dur`, where `ph_seq` is the phoneme sequence and `ph_dur` is the phoneme duration sequence in seconds. You must have all corresponding recordings declared by the `name` column in mono, WAV format.

Training from multiple datasets in one model (so that the model is a multi-speaker model) is supported. See `speakers`, `spk_ids` and `use_spk_id` in the configuration schemas.

### Functionalities

Functionalities of acoustic models are defined by their inputs. Acoustic models have three basic and fixed inputs: phoneme sequence, phoneme duration sequence and F0 (pitch) sequence. There are three categories of additional inputs (control parameters):

- speaker IDs: if your acoustic model is a multi-speaker model, you can use different speaker in the same model, or mix their timbre and style.
- variance parameters: these curve parameters are features extracted from the recordings, and can control the timbre and style of the singing voice. See `use_energy_embed` and `use_breathiness_embed` in the configuration schemas. Please note that variance parameters **do not have default values**, so they are usually obtained from the variance model at inference time.
- transition parameters: these values represent the transition of the mel-spectrogram, and are obtained by enabling data augmentation. They are scalars at training time and sequences at inference time. See `augmentation_args`, `use_key_shift_embed` and `use_speed_embed` in the configuration schemas.

## Overview: training variance models

A variance model takes high-level music information as input, including phoneme sequence, word division, word durations and music scores. The outputs of a variance model may include phoneme durations, pitch curve and other control parameters that will be consumed by acoustic models. Briefly speaking, a variance model works as an auxiliary tool (so-called _automatic parameter generator_) for the acoustic models.

### Datasets

To train a variance model, you must have all the required attributes listed in the following table in your transcriptions.csv according to the functionalities enabled.

|                                | name | ph_seq | ph_dur | ph_num | note_seq | note_dur |
|:------------------------------:|:----:|:------:|:------:|:------:|:--------:|:--------:|
|  phoneme duration prediction   |  ✓   |   ✓    |   ✓    |   ✓    |          |          |
|        pitch prediction        |  ✓   |   ✓    |   ✓    |        |    ✓     |    ✓     |
| variance parameters prediction |  ✓   |   ✓    |   ✓    |        |          |          |

The recommended way of building a variance dataset is to extend an acoustic dataset. You may have all the recordings prepared like the acoustic dataset as well, or [use DS files in your variance datasets](#build-variance-datasets-with-ds-files).

Variance models support multi-speaker settings like acoustic models do.

### Functionalities

Functionalities of variance models are defined by their outputs. There are three main prediction modules that can be enabled/disable independently:

- Duration Predictor: predicts the phoneme durations. See `predict_dur` in the configuration schemas.
- Pitch Predictor: predicts the pitch curve. See `predict_pitch` in the configuration schemas.
- Multi-Variance Predictor: jointly predicts other variance parameters. See `predict_energy` and `predict_breathiness` in the configuration schemas.

There may be some mutual influence between the modules above when they are enabled together. See [mutual influence between variance modules](#mutual-influence-between-variance-modules) for more details.

## Build variance datasets with DS files

By default, the variance binarizer loads attributes from transcriptions.csv and searches for recording files (*.wav) to extract features and parameters. These attributes and parameters also exist in DS files, which are normally used for inference. This section introduces the required settings and important notes to build a variance dataset from DS files.

First of all, you should edit your configuration file to enable loading from DS files:

```yaml
binarization_args:
  prefer_ds: true  # prefer loading from DS files
```

Then you should prepare some DS files which are properly segmented. If you export DS files with OpenUTAU for DiffSinger, the DS files are already segmented according to the spaces between notes. You should put these DS files in a folder named `ds` in your raw dataset directory (besides the `wavs` folder).

The DS files should also use the same dictionary as that of your target model. The attributes required vary from your target functionalities, as listed below:

|        attribute name        | required by duration prediction | required by pitch prediction | required by variance parameters prediction | previous source | current source |
|:----------------------------:|:-------------------------------:|:----------------------------:|:------------------------------------------:|:---------------:|:--------------:|
|            `name`            |                ✓                |              ✓               |                     ✓                      |       CSV       |      CSV       |
|           `ph_seq`           |                ✓                |              ✓               |                     ✓                      |       CSV       |     DS/CSV     |
|           `ph_dur`           |                ✓                |              ✓               |                     ✓                      |       CSV       |     DS/CSV     |
|           `ph_num`           |                ✓                |                              |                                            |       CSV       |     DS/CSV     |
|          `note_seq`          |                                 |              ✓               |                                            |       CSV       |     DS/CSV     |
|          `note_dur`          |                                 |              ✓               |                                            |       CSV       |     DS/CSV     |
|           `f0_seq`           |                ✓                |              ✓               |                     ✓                      |       WAV       |     DS/WAV     |
| `energy`, `breathiness`, ... |                                 |                              |                     ✓                      |       WAV       |     DS/WAV     |

This means you only need one column in transcriptions.csv, the `name` column, to declare all DS files included in the dataset. The name pattern can be:

- Full name: `some-name` will firstly match the first segment in `some-name.ds`.
- Name with index: `some-name#0` and `some-name#1` will match segment 0 and segment 1 in `some-name.ds` if there are no match with full name.

Though not recommended, the binarizer will still try to load attributes from transcriptions.csv or extract parameters from recordings if there are no matching DS files. In this case the full name matching logic is applied (the same as the normal binarization process).

## Choosing variance parameters

Variance parameters are a type of parameters that are significantly related to singing styles and emotions, have no default values and need to be predicted by the variance models. Choosing the proper variance parameters can obtain more controllability and expressiveness for your singing models. In this section, we are only talking about **narrowly defined variance parameters**, which are variance parameters except the pitch.

### Supported variance parameters

#### Energy

> WARNING
>
> This parameter is no longer recommended in favor of the new voicing parameter. The latter are less coupled with breathiness than energy.

Energy is defined as the RMS curve of the singing, in dB, which can control the strength of voice to a certain extent.

#### Breathiness

Breathiness is defined as the RMS curve of the aperiodic part of the singing, in dB, which can control the power of the air and unvoiced consonants in the voice.

#### Voicing

Voicing is defined as the RMS curve of the harmonic part of the singing, in dB, which can control the power of the harmonics in vowels and voiced consonants in the voice.

#### Tension

Tension is mostly related to the ratio of the base harmonic to the full harmonics, which can be used to control the strength and timbre of the voice. The ratio is calculated as
$$
r = \frac{\text{RMS}(H_{full}-H_{base})}{\text{RMS}(H_{full})}
$$
where $H_{full}$ is the full harmonics and $H_{base}$ is the base harmonic. The ratio is then mapped to the final domain via the inverse function of Sigmoid, that
$$
T = \log{\frac{r}{1-r}}
$$
where $T$ is the tension value.

### Principles of choosing multiple parameters

#### Energy, breathiness and voicing

These three parameters should **NOT** be enabled together. Energy is the RMS of the full waveform, which is the composition of the harmonic part and the aperiodic part. Therefore, these three parameters are coupled with each other.

#### Energy, voicing and tension

When voicing (or energy) is enabled, it almost fixes the loudness. However, tension sometimes rely on the implicitly predicted loudness for more expressiveness, because when a person sings with higher tension, he/she always produces louder voice. For this reason, some people may find their models or datasets _less natural_ with tension control. To be specific, changing tension will change the timbre but keep the loudness, and changing voicing (or energy) will change the loudness but keep the timbre. This behavior can be suitable for some, but not all datasets and users. Therefore, it is highly recommended for everyone to conduct some experiments on the actual datasets used to train the model.

## Mutual influence between variance modules

In some recent experiments and researches, some mutual influence between the modules of variance models has been found. In practice, being aware of the influence and making use of it can improve accuracy and avoid instability of the model.

### Influence on the duration predictor

The duration predictor benefits from its downstream modules, like the pitch predictor and the variance predictor.

The experiments were conducted on both manually refined datasets and automatically labeled datasets, and with pitch predictors driven by both base pitch and melody encoder. All the results have shown that when either of the pitch predictor and the variance predictor is enabled together with the duration predictor, its rhythm correctness and duration accuracy significantly outperforms those of a solely trained duration predictor.

Possible reason for this difference can be the lack of information carried by pure phoneme duration sequences, which may not fully represent the phoneme features in the real world. With the help of frame-level feature predictors, the encoder learns more knowledge about the voice features related to the phoneme types and durations, thus making the duration predictor produce better results.

### Influence on frame-level feature predictors

Frame-level feature predictors, including the pitch predictor and the variance predictor, have better performance when trained without enabling the duration predictor.

The experiments found that when the duration predictor is enabled, the pitch accuracy drops and the dynamics of variance parameters sometimes become unstable. And it has nothing to do with the gradients from the duration predictor, because applying a scale factor on the gradients does not make any difference even if the gradients are completely cut off.

Possible reason for this phenomenon can be the lack of direct phoneme duration input. When the duration predictor is enabled, the model takes in word durations instead of phoneme durations; when there is no duration predictor together, the phoneme duration sequence is directly taken in and passed through the attention-based linguistic encoder. With direct modeling on the phoneme duration, the frame-level predictors can have a better understanding of the context, thus producing better results.

Another set of experiments showed that there is no significant influence between the pitch predictor and the variance predictor. When they are enabled together without the duration predictor, both can converge well and produce satisfactory results. No conclusion can be drawn on this issue, and it can depend on the dataset.

### Suggested procedures of training variance models

According to the experiment results and the analysis above, the suggested procedures of training a set of variance models are listed below:

1. Train the duration predictor together with the variance predictor, and discard the variance predictor part.
2. Train the pitch predictor and the variance predictor separately or together.
3. If interested, compare across different combinations in step 2 and choose the best.

## Feature extraction

Feature extraction is the process of extracting low-level features from the recordings, which are needed as inputs for the acoustic models, or as outputs for the variance models.

### Pitch extraction

A pitch extractor estimates pitch (F0 sequence) from given recordings. F0 (fundamental frequency) is one of the most important components of singing voice that is needed by both acoustic models and variance models.

```yaml
pe: parselmouth  # pitch extractor type
pe_ckpt: checkpoints/xxx/model.pt  # pitch extractor model path (if it requires any)
```

#### Parselmouth

[Parselmouth](https://github.com/YannickJadoul/Parselmouth) is the default pitch extractor in this repository. It is based on DSP algorithms, runs fast on CPU and can get accurate F0 on clean and normal recordings.

To use parselmouth, simply include the following line in your configuration file:

```yaml
pe: parselmouth
```

#### RMVPE (recommended)

[RMVPE](https://github.com/Dream-High/RMVPE) (Robust Model for Vocal Pitch Estimation) is the state-of-the-art NN-based pitch estimation model for singing voice. It runs slower than parselmouth, consumes more memory, however uses CUDA to accelerate computation (if available) and produce better results on noisy recordings and edge cases.

To enable RMVPE, download its pre-trained checkpoint from [here](https://github.com/yxlllc/RMVPE/releases), extract it into the `checkpoints/` folder and edit the configuration file:

```yaml
pe: rmvpe
pe_ckpt: checkpoints/rmvpe/model.pt
```

#### Harvest

Harvest (Harvest: A high-performance fundamental frequency estimator from speech signals) is the recommended pitch extractor from Masanori Morise's [WORLD](https://github.com/mmorise/World), a free software for high-quality speech analysis, manipulation and synthesis. It is a state-of-the-art algorithmic pitch estimator designed for speech, but has seen use in singing voice synthesis. It runs the slowest compared to the others, but provides very accurate F0 on clean and normal recordings compared to parselmouth.

To use Harvest, simply include the following line in your configuration file:

```yaml
pe: harvest
```

**Note:** It is also recommended to change the F0 detection range for Harvest with accordance to your dataset, as they are hard boundaries for this algorithm and the defaults might not suffice for most use cases. To change the F0 detection range, you may include or edit this part in the configuration file:

```yaml
f0_min: 65  # Minimum F0 to detect
f0_max: 800  # Maximum F0 to detect
```

### Harmonic-noise separation

Harmonic-noise separation is the process of separating the harmonic part and the aperiodic part of the singing voice. These parts are the fundamental components for variance parameters including breathiness, voicing and tension to be calculated from.

#### WORLD

This algorithm uses Masanori Morise's [WORLD](https://github.com/mmorise/World), a free software for high-quality speech analysis, manipulation and synthesis. It uses CPU (no CUDA required) but runs relatively slow.

To use WORLD, simply include the following line in your configuration file:

```yaml
hnsep: world
```

#### Vocal Remover (recommended)

Vocal Remover (VR) is originally a popular NN-based algorithm for music source separation that removes the vocal part from the music. This repository uses a specially trained model for harmonic-noise separation. VR extracts much cleaner harmonic parts, utilizes CUDA to accelerate computation (if available) and runs much faster than WORLD. However, it consumes more memory and should not be used with too many parallel workers.

To enable VR, download its pre-trained checkpoint from [here](https://github.com/yxlllc/vocal-remover/releases), extract it into the `checkpoints/` folder and edit the configuration file:

```yaml
hnsep: vr
hnsep_ckpt: checkpoints/vr/model.pt
```

## Shallow diffusion

Shallow diffusion is a mechanism that can improve quality and save inference time for diffusion models that was first introduced in the original DiffSinger [paper](https://arxiv.org/abs/2105.02446). Instead of starting the diffusion process from purely gaussian noise as classic diffusion does, shallow diffusion adds a shallow gaussian noise on a low-quality results generated by a simple network (which is called the auxiliary decoder) to skip many unnecessary steps from the beginning. With the combination of shallow diffusion and sampling acceleration algorithms, we can get better results under the same inference speed as before, or achieve higher inference speed without quality deterioration.

Currently, acoustic models in this repository support shallow diffusion. The main switch of shallow diffusion is `use_shallow_diffusion` in the configuration file, and most arguments of shallow diffusion can be adjusted under `shallow_diffusion_args`. See [Configuration Schemas](ConfigurationSchemas.md) for more details.

### Train full shallow diffusion models from scratch

To train a full shallow diffusion model from scratch, simply introduce the following settings in your configuration file:

```yaml
use_shallow_diffusion: true
K_step: 400  # adjust according to your needs
K_step_infer: 400  # should be <= K_step
```

Please note that when shallow diffusion is enabled, only the last $K$ diffusion steps will be trained. Unlike classic diffusion models which are trained on full steps, the limit of `K_step` can make the training more efficient. However, `K_step` should not be set too small because without enough diffusion depth (steps), the low-quality auxiliary decoder results cannot be well refined. 200 ~ 400 should be the proper range of `K_step`.

The auxiliary decoder and the diffusion decoder shares the same linguistic encoder, which receives gradients from both the decoders. In some experiments, it was found that gradients from the auxiliary decoder will cause mismatching between the encoder and the diffusion decoder, resulting in the latter being unable to produce reasonable results. To prevent this case, a configuration item called `aux_decoder_grad` is introduced to apply a scale factor on the gradients from the auxiliary decoder during training. To adjust this factor, introduce the following in the configuration file:

```yaml
shallow_diffusion_args:
  aux_decoder_grad: 0.1  # should not be too high
```

### Train auxiliary decoder and diffusion decoder separately

Training a full shallow diffusion model can consume more memory because the auxiliary decoder is also in the training graph. In limited situations, the two decoders can be trained separately, i.e. train one decoder after another.

**STEP 1: train the diffusion decoder**

In the first stage, the linguistic encoder and the diffusion decoder is trained together, while the auxiliary decoder is left unchanged. Edit your configuration file like this:

```yaml
use_shallow_diffusion: true  # make sure the main option is turned on
shallow_diffusion_args:
  train_aux_decoder: false  # exclude the auxiliary decoder from the training graph
  train_diffusion: true  # train diffusion decoder as normal
  val_gt_start: true  # should be true because the auxiliary decoder is not trained yet
```

Start training until `max_updates` is reached, or until you get satisfactory results on the TensorBoard.

**STEP 2: train the auxiliary decoder**

In the second stage, the auxiliary decoder is trained besides the linguistic encoder and the diffusion decoder. Edit your configuration file like this:

```yaml
shallow_diffusion_args:
  train_aux_decoder: true
  train_diffusion: false  # exclude the diffusion decoder from the training graph
lambda_aux_mel_loss: 1.0  # no more need to limit the auxiliary loss
```

Then you should freeze the encoder to prevent it from getting updates. This is because if the encoder changes, it no longer matches with the diffusion decoder, thus making the latter unable to produce correct results again. Edit your configuration file:

```yaml
freezing_enabled: true
frozen_params:
  - model.fs2  # the linguistic encoder
```

You should also manually reset your learning rate scheduler because this is a new training process for the auxiliary decoder. Possible ways are:

1. Rename the latest checkpoint to `model_ckpt_steps_0.ckpt` and remove the other checkpoints from the directory.
2. Increase the initial learning rate (if you use a scheduler that decreases the LR over training steps) so that the auxiliary decoder gets proper learning rate.

Additionally, `max_updates` should be adjusted to ensure enough training steps for the auxiliary decoder.

Once you finished the configurations above, you can resume the training. The auxiliary decoder normally does not need many steps to train, and you can stop training when you get stable results on the TensorBoard. Because this step is much more complicated than the previous step, it is recommended to run some inference to verify if the model is trained properly after everything is finished.

### Add shallow diffusion to classic diffusion models

Actually, all classic DDPMs have the ability to be "shallow". If you want to add shallow diffusion functionality to a former classic diffusion model, the only thing you need to do is to train an auxiliary decoder for it.

Before you start, you should edit the configuration file to ensure that you use the same datasets, and that you do not remove or add any of the functionalities of the old model. Then you can configure the old checkpoint in your configuration file:

```yaml
finetune_enabled: true
finetune_ckpt_path: xxx.ckpt  # path to your old checkpoint
finetune_ignored_params: []  # do not ignore any parameters
```

Then you can follow the instructions in STEP 2 of the [previous section](#add-shallow-diffusion-to-classic-diffusion-models) to finish your training.

## Performance tuning

This section is about accelerating training and utilizing hardware.

### Data loader and batch sampler

The data loader loads data pieces from the binary dataset, and the batch sampler forms batches according to data lengths.

To configure the data loader, edit your configuration file:

```yaml
ds_workers: 4  # number of DataLoader workers
dataloader_prefetch_factor: 2  # load data in advance
```

To configure the batch sampler, edit your configuration file:

```yaml
sampler_frame_count_grid: 6  # lower value means higher speed but less randomness
```

For more details of the batch sampler algorithm and this configuration key, see [sampler_frame_count_grid](ConfigurationSchemas.md#sampler_frame_count_grid).

### Automatic mixed precision

Enabling automatic mixed precision (AMP) can accelerate training and save GPU memory. DiffSinger have adapted the latest version of PyTorch Lightning for AMP functionalities.

By default, the training runs in FP32 precision. To enable AMP, edit your configuration file:

```yaml
pl_trainer_precision: 16-mixed  # FP16 precision
```

or

```yaml
pl_trainer_precision: bf16-mixed  # BF16 precision
```

For more precision options, please check out the [official documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision).

### Training on multiple GPUs

Using distributed data parallel (DDP) can divide training tasks to multiple GPUs and synchronize gradients and weights between them. DiffSinger have adapted the latest version of PyTorch Lightning for DDP functionalities.

By default, the trainer will utilize all CUDA devices defined in the `CUDA_VISIBLE_DEVICES` environment variable (empty means using all available devices). If you want to specify which GPUs to use, edit your configuration file:

```yaml
pl_trainer_devices: [0, 1, 2, 3]  # use the first 4 GPUs defined in CUDA_VISIBLE_DEVICES
```

Please note that `max_batch_size` and `max_batch_frames` are values for **each** GPU.

By default, the trainer uses NCCL as the DDP backend. If this gets stuck on your machine, try disabling P2P first via

```yaml
nccl_p2p: false  # disable P2P in NCCL
```

Or if your machine does not support NCCL, you can switch to Gloo instead:

```yaml
pl_trainer_strategy:
  name: ddp                    # must manually choose a strategy instead of 'auto'
  process_group_backend: gloo  # however, it has a lower performance than NCCL
```

### Gradient accumulation

Gradient accumulation means accumulating losses for several batches before each time the weights are updated. This can simulate a larger batch size with a lower GPU memory cost.

By default, the trainer calls `backward()` each time the losses are calculated through one batch of data. To enable gradient accumulation, edit your configuration file:

```yaml
accumulate_grad_batches: 4  # the actual batch size will be 4x.
```

Please note that enabling gradient accumulation will slow down training because the losses must be calculated for several times before the weights are updated (1 update to the weights = 1 actual training step).

## Optimizers and learning rate schedulers

The optimizer and the learning rate scheduler can take an important role in the training process. DiffSinger uses a flexible configuration logic for these two modules.

### Basic configurations

The optimizer and learning rate scheduler used during training can be configured by their full class name and keyword arguments in the configuration file. Take the following as an example for the optimizer:

```yaml
optimizer_args:
  optimizer_cls: torch.optim.AdamW  # class name of optimizer
  lr: 0.0004
  beta1: 0.9
  beta2: 0.98
  weight_decay: 0
```

and for the learning rate scheduler:

```yaml
lr_scheduler_args:
  scheduler_cls: torch.optim.lr_scheduler.StepLR  # class name of learning rate schedule
  warmup_steps: 2000
  step_size: 50000
  gamma: 0.5
```

Note that `optimizer_args` and `lr_scheduler_args` will be filtered by needed parameters and passed to `__init__` as keyword arguments (`kwargs`) when constructing the optimizer and scheduler. Therefore, you could specify all arguments according to your need in the configuration file to directly control the behavior of optimization and LR scheduling. It will also tolerate parameters existing in the configuration but not needed in `__init__`.

Also, note that the LR scheduler performs scheduling on the granularity of steps, not epochs.

The special case applies when a tuple is needed in `__init__`: `beta1` and `beta2` are treated separately and form a tuple in the code. You could try to pass in an array instead. (And as an experiment, AdamW does accept `[beta1, beta2]`). If there is another special treatment required, please submit an issue.

For PyTorch built-in optimizers and LR schedulers, see official [documentation](https://pytorch.org/docs/stable/optim.html) of the `torch.optim` package. If you found other optimizer and learning rate scheduler useful, you can raise a topic in [Discussions](https://github.com/openvpi/DiffSinger/discussions), raise [Issues](https://github.com/openvpi/DiffSinger/issues) or submit [PRs](https://github.com/openvpi/DiffSinger/pulls) if it introduces new codes or dependencies.

### Composite LR schedulers

Some LR schedulers like `SequentialLR` and `ChainedScheduler` may use other schedulers as arguments. Besides built-in types, there is a special design to configure these scheduler objects. See the following example.

```yaml
lr_scheduler_args:
  scheduler_cls: torch.optim.lr_scheduler.SequentialLR
  schedulers:
  - cls: torch.optim.lr_scheduler.ExponentialLR
    gamma: 0.5
  - cls: torch.optim.lr_scheduler.LinearLR
  - cls: torch.optim.lr_scheduler.MultiStepLR
    milestones:
    - 10
    - 20
  milestones:
  - 10
  - 20
```

The LR scheduler objects will be recursively construct objects if `cls` is present in sub-arguments. Please note that `cls` must be a scheduler class because this is a special design.

**WARNING:** Nested `SequentialLR` and `ChainedScheduler` have unexpected behavior. **DO NOT** nest them. Also, make sure the scheduler is _chainable_ before using it in `ChainedScheduler`.

## Fine-tuning and parameter freezing

### Fine-tuning from existing checkpoints

By default, the training starts from a model from scratch with randomly initialized parameters. However, if you already have some pre-trained checkpoints, and you need to adapt them to other datasets with their functionalities unchanged, fine-tuning may save training steps and time. In general, you need to add the following structure into the configuration file:

```yaml
# take acoustic models as an example
finetune_enabled: true  # the main switch to enable fine-tuning
finetune_ckpt_path: checkpoints/pretrained/model_ckpt_steps_320000.ckpt  # path to your pre-trained checkpoint
finetune_ignored_params:  # prefix rules to exclude specific parameters when loading the checkpoints
  - model.fs2.encoder.embed_tokens  # in case when the phoneme set is changed
  - model.fs2.txt_embed  # same as above
  - model.fs2.spk_embed  # in case when the speaker set is changed
finetune_strict_shapes: true  # whether to raise an error when parameter shapes mismatch
```

For the pre-trained checkpoint, it must be a file saved with `torch.save`, containing a `dict` object and a `state_dict` key, like the following example:

```json5
{
  "state_dict": {
    "model.fs2.txt_embed": null,  // torch.Tensor
    "model.fs2.pitch_embed.weight": null,  // torch.Tensor
    "model.fs2.pitch_embed.bias": null,  // torch.Tensor
    // ... (other parameters)
  }
  // ... (other possible keys
}
```

**IMPORTANT NOTES**:

- The pre-trained checkpoint is **loaded only once** at the beginning of the training experiment. You may interrupt the training at any time, but after this new experiment has saved its own checkpoint, the pre-trained checkpoint will not be loaded again when the training is resumed.
- Only the state dict of the checkpoint will be loaded. The optimizer state in the pre-trained checkpoint will be ignored.
- The parameter name matching is **not strict** when loading the pre-trained checkpoint. This means that missing parameters in the state dict will still be left as randomly initialized, and redundant parameters will be ignored without any warnings and errors. There are cases where the tensor shapes mismatch between the pre-trained state dict and the model - edit `finetune_strict_shapes` to change the behavior when dealing with this.
- Be careful if you want to change the functionalities when fine-tuning. Starting from a checkpoint trained under different functionalities may be even slower than training from scratch.

### Freezing model parameters

Sometimes you want to freeze part of the model during training or fine-tuning to save GPU memory, accelerate the training process or avoid catastrophic forgetting. Parameter freezing may also be useful if you want to add/remove functionalities from pre-trained checkpoints. In general, you need to add the following structure into the configuration file:

```yaml
# take acoustic models as an example
freezing_enabled: true  # main switch to enable parameter freezing
frozen_params:  # prefix rules to freeze specific parameters during training
  - model.fs2.encoder
  - model.fs2.pitch_embed
```

You may interrupt the training and change the settings above at any time. Sometimes this will cause mismatching optimizer state - and it will be discarded silently.
