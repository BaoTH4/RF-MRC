# RF-MRC
In this project, I try to reprogram the model suggested in the paper: "Self Question-answering: Aspect-based Sentiment Analysis by Role
Flipped Machine Reading Comprehension" (Link: (https://aclanthology.org/2021.findings-emnlp.115.pdf))

Author: Guoxin Yu, Jiwei Li, Ling Luo, Yuxian Meng, Xiang Ao, Qing He

# Environment Configuration
```
python 3.9
transformer 4.17.0
pytorch
sentencepiece 0.1.96
```

# Data preprocess

```
python process_data.py
python make_tokenized_data.py
python make_standard_data.py
```

# Training  model
```
python main.py --mode train
```
