from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config
from utils import Vocab, Tokenizer
import os
import sys
import numpy as np
import pickle
from models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='test')
    # config.sample = True
    config.n_sample_step = 1
    file_path = 'gen_samples.txt'

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size,
        shuffle=False)
    config.mode = 'generate'

    config.one_latent_z = 2# change this
    converation_idx = 377 # change this
    sentence_idx = 1  # change this

    if config.model in VariationalModels:
        solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        
    sentence = [load_pickle(config.sentences_path)[converation_idx][sentence_idx]]
    sentence_length = [[load_pickle(config.sentence_length_path)[converation_idx][sentence_idx]]]
    sentence = [[vocab.sent2id(sen) for sen in sentence]]
    print('The number of conversation: ' + str(len(load_pickle(config.conversation_length_path))))
    print('The conversations length of' + str(converation_idx) + ': ' + str(load_pickle(config.conversation_length_path)[converation_idx]))
    # print(sentence_length)
    # print(np.array(sentence).shape)
    # print(np.array(sentence_length).shape)

    solver.build()
    sample = solver.gen_one_sentence(sentence, sentence_length)

    for input_sent, output_sent in zip(sentence, sample):
        input_sent = '\n'.join([vocab.decode(sent) for sent in input_sent])
        output_sent = '\n'.join([vocab.decode(sent) for sent in output_sent])
        s = '\n'.join(['Input sentence: ' + input_sent,
                       'Generated response: ' + output_sent + '\n'])
        print(s)
