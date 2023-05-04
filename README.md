# Easy-Adapter
### Code for AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks

[![Version](https://img.shields.io/badge/Version-v0.1.0-blue?color=FF8000?color=009922)](https://img.shields.io/badge/Version-v0.1.0-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-pink?color=FF33CC)](https://github.com/huggingface/transformers)

arXiv link: https://arxiv.org/abs/2205.00305
[**Findings of NAACL 2022**](https://2022.naacl.org/)

This code demonstrates how to fine-tune a BERT model based on the Hugging Face Transformers library using adapters.

## Fine-tuning with adapters on the GLUE benchmark
```
bash run_glue_adapter.sh
```

## Fine-tuning with adapters on IMdB task
```
python run_imdb.py
```

If you use this code in your research, please cite the following papers:

```
@article{fu2022adapterbias,
  title={AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks},
  author={Fu, Chin-Lun and Chen, Zih-Ching and Lee, Yun-Ru and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2205.00305},
  year={2022}
}

@inproceedings{chen2023exploring,
  title={Exploring efficient-tuning methods in self-supervised speech models},
  author={Chen, Zih-Ching and Fu, Chin-Lun and Liu, Chih-Ying and Li, Shang-Wen Daniel and Lee, Hung-yi},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages={1120--1127},
  year={2023},
  organization={IEEE}
}
```

This code demonstrates a practical example of using adapters in fine-tuning a BERT model. The code can be adapted to other pre-trained models and NLP tasks.
