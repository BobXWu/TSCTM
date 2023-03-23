# Code for Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning (EMNLP 2022)

[EMNLP 2022](https://aclanthology.org/2022.emnlp-main.176)


## Usage

### 1. Prepare Environment

    python==3.7
    pytorch==1.7.1
    scipy==1.7.3
    scikit-learn==0.23.2
    pyyaml==6.0


### 2. Training

Training without data augmentation:

    python run.py --data_dir data/${dataset} --model TSCTM --num_topic {number of topic}

Training with data augmentation:

    python run.py --data_dir data/${dataset} --model TSCTM_aug --num_topic {number of topic}


After training, outputs are in `./output/{dataset}/`, including topic words, topic distributions of short texts, and topic-word distribution matrix.


### 3. Evaluation

**topic coherence**: [coherence score](https://github.com/dice-group/Palmetto)


**topic diversity**:

    python utils/TU.py --data_path {path of topic word file}


## Citation

If you want to use our code, please cite as

    @inproceedings{wu-etal-2022-mitigating,
        title = "Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning",
        author = "Wu, Xiaobao  and
          Luu, Anh Tuan  and
          Dong, Xinshuai",
        booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
        month = dec,
        year = "2022",
        address = "Abu Dhabi, United Arab Emirates",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.emnlp-main.176",
        pages = "2748--2760",
    }
