# Parameter-efficient Fine-tuning in NLP

This code demonstrates how to fine-tune a BERT model based on the Hugging Face Transformers library using adapters. Adapters are a parameter-efficient way to fine-tune a pre-trained language model for a specific NLP task.

The code was written by [Zih-Ching Chen](https://github.com/virginiakm1988).

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
