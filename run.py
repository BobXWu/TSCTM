import os
import argparse
import scipy
import torch
import yaml
from torch.utils.data import DataLoader
from utils import data_utils
from utils.Data import TextData
from runners.Runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--en1_units', type=int, default=100)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=1)

    args = parser.parse_args()
    return args


def save_theta(model_runner, train_dataset, use_aug, output_prefix):

    train_theta = model_runner.test(train_dataset.train_bow)
    scipy.sparse.save_npz('{}_theta_train.npz'.format(output_prefix), scipy.sparse.csr_matrix(train_theta))

    if use_aug:
        scipy.sparse.save_npz('{}_theta_train.npz'.format(output_prefix), scipy.sparse.csr_matrix(train_theta))

        for i, bow in enumerate(train_dataset.contrast_bow_list):
            contrast_theta = model_runner.test(bow)
            scipy.sparse.save_npz('{}_theta_{}.npz'.format(output_prefix, train_dataset.aug_option_list[i]), scipy.sparse.csr_matrix(contrast_theta))
    else:
        scipy.sparse.save_npz('{}_theta_train.npz'.format(output_prefix), scipy.sparse.csr_matrix(train_theta))


def main():
    config = parse_args()
    data_utils.update_args(config, f"configs/{config.model}.yaml")
    print("===>Info: args: \n", yaml.dump(vars(config), default_flow_style=False))

    aug_option_list = None
    if config.use_aug:
        aug_option_list = ['contextual0.3', 'wordnet0.3']
        aug_option_list.sort()
        print("===>Info: aug_option_list: ", aug_option_list)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = TextData(config.data_dir, device, aug_option_list, config.use_aug)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    config.vocab_size = len(train_dataset.vocab)

    # Training
    model_runner = Runner(config, device)
    beta = model_runner.train(train_loader)

    # Save output
    dataset_name = os.path.basename(config.data_dir)
    output_prefix = f'output/{dataset_name}/{config.model}_K{config.num_topic}_{config.test_index}th'

    data_utils.make_dir(os.path.dirname(output_prefix))
    topic_str_list = data_utils.print_topic_words(beta, train_dataset.vocab, config.num_top_word)
    data_utils.save_text(topic_str_list, f'{output_prefix}_T{config.num_top_word}')

    save_theta(model_runner, train_dataset, config.use_aug, output_prefix)
    scipy.sparse.save_npz(f'{output_prefix}_beta.npz', scipy.sparse.csr_matrix(beta))


if __name__ == '__main__':
    main()
