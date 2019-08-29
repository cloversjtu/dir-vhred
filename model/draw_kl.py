from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config
from utils import Vocab, Tokenizer
import os
import pickle
from models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='valid')

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    for i in range(50):
        check_file = str(i+1) + '.pkl'
        config.checkpoint = os.path.join(config.eval_checkpoint, check_file)
        data_loader = get_loader(
            sentences=load_pickle(config.sentences_path),
            conversation_length=load_pickle(config.conversation_length_path),
            sentence_length=load_pickle(config.sentence_length_path),
            vocab=vocab,
            batch_size=config.batch_size)

        if config.model in VariationalModels:
            solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
            solver.build()
            solver.evaluate()
        else:
            solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
            solver.build()
            solver.test()