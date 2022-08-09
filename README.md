# Team FeelsGood: MuSe-Stress 2022, LSTM Regressor + Transformer Encoder

[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress)


## Introduction

This git contains the MuSe 2022 participating team FeelsGood output. We added a Transformer Encoder model to the existing Baseline LSTM model and organized it to be compatible with the existing code as much as possible. For details about competition, please see the [Baseline Paper](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress).

If you would like to see our approach and its results please see [Our paper](https://willbeupdated.com) 

The followings are the deliverables from this project.

* New extracted feature, Pose : [Download](https://willbeupdated.com)
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

The MuSe2022 baseline paper is available as a preprint [here](https://www.researchgate.net/publication/359875358_The_MuSe_2022_Multimodal_Sentiment_Analysis_Challenge_Humor_Emotional_Reactions_and_Stress)

```bibtex
@inproceedings{Christ22-TM2,
  title={The MuSe 2022 Multimodal Sentiment Analysis Challenge: Humor, Emotional Reactions, and Stress},
  author={Christ, Lukas and Amiriparian, Shahin and Baird, Alice and Tzirakis, Panagiotis and Kathan, Alexander and Müller, Niklas and Stappen, Lukas and Meßner, Eva-Maria and König, Andreas and Cowen, Alan and Cambria, Erik and Schuller, Bj\"orn W. },
  booktitle={Proceedings of the 3rd Multimodal Sentiment Analysis Challenge},
  year={2022},
  address = {Lisbon, Portugal},
  publisher = {Association for Computing Machinery},
  note = {co-located with ACM Multimedia 2022, to appear}
}

```

MuSe 2021 baseline paper:

```bibtex
@incollection{stappen2021muse,
  title={The MuSe 2021 multimodal sentiment analysis challenge: sentiment, emotion, physiological-emotion, and stress},
  author={Stappen, Lukas and Baird, Alice and Christ, Lukas and Schumann, Lea and Sertolli, Benjamin and Messner, Eva-Maria and Cambria, Erik and Zhao, Guoying and Schuller, Bj{\"o}rn W},
  booktitle={Proceedings of the 2nd on Multimodal Sentiment Analysis Challenge},
  pages={5--14},
  year={2021}
}

```
