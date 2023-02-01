import os
import argparse
import yaml
import numpy as np


def print_topic_words(beta, vocab, num_top_word=15):
    
    topic_str_list = []
    for i, topic_dist in enumerate(beta):
        topic_words = np.asarray(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        # print('Topic {}: {}'.format(i + 1, topic_str))
    return topic_str_list


def update_args(args, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
        if config:
            args = vars(args)
            args.update(config)
            args = argparse.Namespace(**args)
    print("===>Info: use setting in file {}.".format(path))


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text.strip() + '\n')


def split_text_word(texts):
    texts = [text.split() for text in texts]
    return texts
