# Team FeelsGood: MuSe-Stress 2022, LSTM Regressor + Transformer Encoder

[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress)


## Introduction

This git contains the MuSe 2022 participating team FeelsGood output. We added a Transformer Encoder model to the existing Baseline LSTM model and organized it to be compatible with the existing code as much as possible. For details about competition, please see the [Baseline Paper](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress).

If you would like to see our approach and its results please see [Our paper](https://dl.acm.org/doi/pdf/10.1145/3551876.3554807) 

The followings are the deliverables from this project.

* New extracted feature, Pose : [Download](https://drive.google.com/drive/folders/19jevZIZt51tJ67OtQyumMaHFrhRL4AJP?usp=share_link)
* Trained models : [Unimodal](https://willbeupdated.com) [Multimodal](https://willbeupdated.com)


## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH``). 

You can then e.g. run the unimodal baseline reproduction calls in the ``*.sh`` file provided for each sub-challenge.

## Settings
The ``main.py`` script is used for training and evaluating models. Most important options:
* ``--model_type``: choose either `LSTM` or `Transformer`
* ``--task``: choose either `humor`, `reaction` or `stress` 
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for eGeMAPS features).
* Options defining the model architecture: ``d_rnn``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``
* Specific parameters for MuSe-Stress: ``emo_dim`` (``valence`` or ``physio-arousal``), ``win_len`` and ``hop_len`` for segmentation.

For more details, please see the ``parse_args()`` method in ``main.py``. 


## Citation:
```bibtex
@inproceedings{10.1145/3551876.3554807,
author = {Park, Ho-min and Yun, Ilho and Kumar, Ajit and Singh, Ankit Kumar and Choi, Bong Jun and Singh, Dhananjay and De Neve, Wesley},
title = {Towards Multimodal Prediction of Time-Continuous Emotion Using Pose Feature Engineering and a Transformer Encoder},
year = {2022},
isbn = {9781450394840},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3551876.3554807},
doi = {10.1145/3551876.3554807},
booktitle = {Proceedings of the 3rd International on Multimodal Sentiment Analysis Workshop and Challenge},
pages = {47–54},
numpages = {8},
keywords = {multimodal fusion, emotion detection, multimodal sentiment analysis, human pose},
location = {Lisboa, Portugal},
series = {MuSe' 22}
}
```
