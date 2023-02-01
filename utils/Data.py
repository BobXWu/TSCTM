import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset
from utils import data_utils


class TextData(Dataset):
    def __init__(self, data_dir, device, aug_option_list=None, use_aug=False):
        name = 'train_texts.txt'
        self.train_texts = data_utils.read_text(os.path.join(data_dir, '{}'.format(name)))
        train_size = len(self.train_texts)

        self.use_aug = use_aug
        self.aug_option_list = aug_option_list

        vectorizer = CountVectorizer()
        self.train_bow = vectorizer.fit_transform(self.train_texts).toarray().astype('float32')

        self.train_vocab = vectorizer.get_feature_names_out()

        if aug_option_list:
            print('===>Info: reading augmentation data...')

            self.num_contrast = len(aug_option_list)
            self.aug_texts_list = list()
            for aug_option in aug_option_list:
                aug_text_path = os.path.join(data_dir, '{}_{}'.format(name, aug_option))
                print('===>reading {}'.format(aug_text_path))
                self.aug_texts_list.append(data_utils.read_text(aug_text_path))

            self.combined_train_texts = np.concatenate((np.asarray(self.train_texts), np.asarray(self.aug_texts_list).flatten()))
            combined_train_bow = vectorizer.fit_transform(self.combined_train_texts).toarray().astype('float32')

            if self.use_aug:
                self.train_bow = combined_train_bow[:train_size]
                self.contrast_bow_list = np.array_split(combined_train_bow[train_size:], self.num_contrast)
                self.contrast_bow_list = [torch.tensor(bow).to(device) for bow in self.contrast_bow_list]

            else:
                self.train_bow = combined_train_bow

        self.train_bow = torch.tensor(self.train_bow).to(device)
        self.vocab = vectorizer.get_feature_names_out()

    def __len__(self):
        return len(self.train_bow)

    def __getitem__(self, idx):
        if self.use_aug:
            return [self.train_bow[idx]] + [bow[idx] for bow in self.contrast_bow_list]
        else:
            return self.train_bow[idx]
