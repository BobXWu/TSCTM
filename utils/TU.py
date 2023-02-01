import os
import argparse
import numpy as np
import data_utils
from sklearn.feature_extraction.text import CountVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args


def TU_eva(texts, verbose=False):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TU = 0.0
    TF = counter.sum(axis=0)
    cnt = TF * (counter > 0)

    # # output most frequent words
    if verbose:
        word_index_dict = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))
        for index in TF.argsort()[-20:][::-1]:
            print('{:10s}: {}'.format(word_index_dict[index], TF[index]))
        print()

    for i in range(K):
        TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
    TU /= K

    return TU


if __name__ == "__main__":
    args = parse_args()
    texts = data_utils.read_text(args.data_path)
    TU = TU_eva(texts, verbose=args.verbose)
    print("===>TU: {:5f}".format(TU))
