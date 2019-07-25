import argparse
import logging
import gensim
import os
from process_utils import load_poems
import logging.config
import yaml
from poem_rank import  compute_poem_rank
import process_utils
import json
import random
random.seed(10)
import numpy as np
import pickle
from constants import *
from gensim.models import Word2Vec
from bpe import Encoder


def setup_logging(default_path='logging.yaml', default_level=logging.INFO,
                  env_key='LOG_CFG'):

    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return logging.getLogger(__name__)


def pipe_run(poem_path, save_path, min_count=5, embed_size=512):

    poems = load_poems(poem_path)
    logging.info('read %d poems ' % len(poems))

    logging.info('BPE based vocab mining.....')
    bpe_encoder = Encoder(vocab_size=30000, pct_bpe=1.0, ngram_min=2, ngram_max=2)  # params chosen for demonstration purposes
    bpe_poems = ["".join(s[:-1]) for p in poems for s in p ]
    # print(bpe_poems[:10])
    bpe_encoder.fit(bpe_poems)

    flatten_poems = [[BEGIN_TOKEN]+[SEP if c in {'，','。'} else c for sen in p for c in sen]+[END_TOKEN]
                     for p in poems]

    logging.info('%d training lines' % len(flatten_poems))
    logging.info('training poem embedding model......')

    model = gensim.models.Word2Vec(flatten_poems, size=embed_size,
                                   min_count=min_count, workers=4,
                                   window=7, iter=5, negative=5)

    logging.info('store poem embedding model.......')
    logging.info('%d vocab is stored', len(model.wv.vocab))
    model.save(os.path.join(save_path,"word2vec.model"))
    logging.info('compute swap embedding...')
    compute_embed_swap(poems, bpe_encoder)

    logging.info('compute the phrase rank in all poems')
    phrase_rank = compute_poem_rank(poems, bpe_encoder)
    # print( [(k, phrase_rank[k])for k in sorted(phrase_rank, key=phrase_rank.get, reverse=True)])
    with open(save_path+'phrase_rank.json', 'w', encoding='utf-8') as fp:
        json.dump(phrase_rank, fp)
    logging.info('build the vocabulary....')
    token2id = {v: i+2 for i, v in enumerate(set(model.wv.vocab))}
    token2id[PAD] = 0
    token2id[UNK] = 1
    id2token = {v:k for k,v in token2id.items()}

    with open(save_path+'token2id.json', 'w', encoding='utf-8') as fp:
        json.dump(token2id, fp)
    with open(save_path+'id2token.json', 'w', encoding='utf-8') as fp:
        json.dump(id2token, fp)

    logging.info('vocab size is %d ' % len(token2id))
    logging.info('build embedding table......')
    embed_table = np.zeros((len(token2id), embed_size))
    for index, token in id2token.items():

        if token != UNK and token!= PAD:
            embed_table[token2id[token]] = model.wv[token]
        else:
            embed_table[token2id[token]] = np.random.uniform(size=embed_size)

    logging.info(embed_table.shape)
    np.save(save_path+'embed_table.npy', embed_table)
    logging.info('generate train and val and test......')
    dgen = process_utils.DatasetGen(token2id, phrase_rank, bpe_encoder)
    train_dataset, val_dataset, test_dataset = dgen.gen_train_val_test(poems, size=8000)
    logging.info('size of training %d' % len(train_dataset))
    logging.info('size of validation %d' % len(val_dataset))
    logging.info('size of test %d' % len(test_dataset))
    with open(save_path+'train.pkl', 'wb') as fp:
        pickle.dump(train_dataset, fp)
    with open(save_path+'test.pkl', 'wb') as fp:
        pickle.dump(test_dataset, fp)
    with open(save_path+'val.pkl', 'wb') as fp:
        pickle.dump(val_dataset, fp)


    return bpe_encoder

def compute_embed_swap(poems, bpe_encoder):
    pass



def random_check(bpe_encoder):
    with open('../model/train.pkl', 'rb') as fp:
        train = pickle.load(fp)
    triple = np.random.randint(0, len(train))
    triple = train[triple]
    with open('../model/phrase_rank.json', 'r', encoding='utf-8') as fp:
        phrase_rank = json.load(fp)
    with open('../model/token2id.json', 'r', encoding='utf-8') as fp:
        token2id = json.load(fp)
    dg = process_utils.DatasetGen(token2id, phrase_rank, bpe_encoder)
    keyword = dg.recover(triple[0])
    preceeding_text = dg.recover(triple[1])
    current_line = dg.recover(triple[2])
    print(keyword)
    print(preceeding_text)
    print(current_line)


def check_we():
    model = Word2Vec.load("../model/word2vec.model")
    print(model.most_similar(positive=['故'], negative=[]))
    exit(0)


if __name__ == '__main__':

    check_we()
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path',type=str, help='specify the path that stores the poems',
                        default='../data/poems.txt', metavar='<str>')
    parser.add_argument('-s', '--save', dest='save', type=str, help='save dir', default='../model/')
    args = parser.parse_args()
    bpe_enc = pipe_run(args.path, args.save)
    random_check(bpe_enc)
