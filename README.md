# One Deep Music Representation to Rule Them All?
## A comparative analysis of different representation learning strategies
---

This repository provides the codebase used in the [work](https://github.com/eldrin/MTLMusicRepresentation-PyTorch). This codebase includes all the relavant codes that are used for the experiments: 1) data preprocessing, 2) training, 3) feature extraction, 4) evaluation. The core functionalities are implemented and located under main source directory `musmtl` and scripts for each step of the experimental pipeline is located in `scripts` directory. The evaluation setups (the tracks and splits that are used in the evaluation setup) are stored in `eval/data/` directory. The recommender system models used in the experiments are implemented in [separated repository](https://github.com/eldrin/mf-numba/tree/7f2e5eef3e1a401117c70674cec066b37af8be59), for further maintenance and development. (we're planning keep developing this!)

---
## Dependencies
---
`TBD`

---
## Feature Extraction
---

To extract CNN features from provided model, one should first download the model files from the [dataset page](https://commingso.on). Once the model is stored locally, one then can extract features by calling following command.

```
$python scripts/ext_feat.py model_checkpoints.txt target_audios.txt /path/to/save/outputs/ --no-gpu
```

The `model_checkpoints.txt` should contain the paths for the models you want to deploy for the extraction (one model per line). The `target_audios.txt` file is another text file that lists the audio files to be processed. The third argument is the path where the output files going to be saved. Final argument, either `--no-gpu` or `--gpu`, is the flag for indicating the computation is executed on GPU or CPU. (If your computing environment is not equiped with any GPU, the process will be forced to be done in CPU)

---
## Reproduction of the Experiment
---
`TDB`

---
## Referencing the Work
---
To use this work for other research projects or paper, please consider citing:

```
@article{DBLP:journals/corr/abs-1802-04051,
  author    = {Jaehun Kim and
               Juli{\'{a}}n Urbano and
               Cynthia C. S. Liem and
               Alan Hanjalic},
  title     = {One Deep Music Representation to Rule Them All? : {A} comparative
               analysis of different representation learning strategies},
  journal   = {CoRR},
  volume    = {abs/1802.04051},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.04051},
  archivePrefix = {arXiv},
  eprint    = {1802.04051},
  timestamp = {Mon, 13 Aug 2018 16:47:30 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1802-04051},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
