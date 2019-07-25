import re
import jieba
import logging
import numpy as np
import json
from constants import *
from keras.preprocessing.sequence import pad_sequences
import jieba.posseg as pseg
from tqdm import *


def load_poems(poem_path):
    poems = []
    with open(poem_path, "r", encoding='utf-8', ) as fp:
        for line in fp.readlines():
            try:

                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        END_TOKEN in content or BEGIN_TOKEN in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = [sen+'。' for sen in re.split(r'[，。]', content.strip())][:-1]
                poems.append(content)

            except ValueError as e:
                pass
    return poems


class DatasetGen(object):

    def __init__(self, token2id, phrase_rank, bpe_encoder):
        self.token2id = token2id
        self.phrase_rank = phrase_rank
        self.id2token = {v:k for k, v in self.token2id.items()}
        self.bpe_encoder = bpe_encoder

    def poem2vec(self):
        pass

    def gen_train_val_test(self, poems, size=2000):
        keywords = list()
        preceeding_texts = list()
        current_lines = list()

        for poem in tqdm(poems):

            preceeding_text = list()
            # print(poem)
            for sen in poem:
                sen = sen[:-1]
                current_line = self.vectorize_sen(sen)
                # pre = preceding_text #self.vectorize_pres(preceding_text)
                keyword = self.get_keyword(sen)
                # print(sen)
                # print([self.id2token[k] for k in keyword])
                # print('-'*10)
                # add into the dataset
                keywords.append([self.token2id[BEGIN_TOKEN]]+keyword+[self.token2id[END_TOKEN]])
                current_lines.append([self.token2id[BEGIN_TOKEN]]+current_line+[self.token2id[END_TOKEN]])
                preceeding_texts.append([self.token2id[BEGIN_TOKEN]]+preceeding_text+[self.token2id[END_TOKEN]])
                # enhance the preceeding text
                if len(preceeding_text) == 0:
                    preceeding_text = current_line
                else:
                    preceeding_text = preceeding_text + [self.token2id[SEP]]+current_line

            # print('*'*10)
            # input('continue.........')

        logging.info('total data size is %d' % len(keywords))

        # X
        keywords = np.asarray(pad_sequences(keywords, maxlen=5,
                                            truncating='pre',
                                            padding='pre'
                                            ))

        preceeding_texts = np.asarray(pad_sequences(preceeding_texts, maxlen=25,
                                                    truncating='pre',
                                                    padding='pre'
                                                    ))
        # Y
        current_lines = np.asarray(pad_sequences(current_lines, maxlen=9,
                                                 truncating='post',
                                                 padding='post'
                                                 ))

        pers = np.random.choice(len(keywords), len(keywords), replace=False)
        preceeding_texts = preceeding_texts[pers]
        keywords = keywords[pers]
        current_lines = current_lines[pers]

        dataset = [[keywords[i], preceeding_texts[i], current_lines[i]] for i in range(len(keywords))]
        train_point = len(dataset)-size*2
        val_point = len(dataset)-size
        train_dataset = dataset[:train_point]
        val_dataset = dataset[train_point:val_point]
        test_dataset = dataset[val_point:]
        return train_dataset, val_dataset, test_dataset

    def is_noun(self, word):
        ps = pseg.cut(word)
        for p, flag in ps:
            if 'n' in flag:
                return True
        else:
            return False

    def compute_score(self, word):
        score = 0.0
        cn = 0
        for c in word:
            if c in self.phrase_rank:
                score += self.phrase_rank[c]
                cn += 1
        if cn == 0:
            return 0.0
        else:
            if self.is_noun(word):
                score *= 10000
            return score/cn



    def get_keyword(self, sen):

        max_seg = None
        max_val = -1
        seg = self.bpe_encoder.tokenize(sen)[1:-1]
        # print(seg)
        for word in seg:
            score = self.compute_score(word)
            if max_val <  score :
                max_val = score
                max_seg = word
            # print(word, score, self.is_noun(word))

        if max_seg is None:
            return list()
        else:
            return self.vectorize_sen(max_seg)

        """
        seg_list = jieba.cut(sen[:-1], cut_all=False)
        max_seg = None
        max_val = -1

        for seg in seg_list:
            if seg in self.phrase_rank and max_val < self.phrase_rank.get(seg):
                max_val = self.phrase_rank.get(seg)
                max_seg = seg

        if max_seg is None:
            return list()
        else:
            return self.vectorize_sen(max_seg)
        """


    def vectorize_sen(self, text):
        return [self.token2id[t] if t in self.token2id.keys() else self.token2id[UNK] for t in text]

    def vectorize_pres(self, pres):
        if len(pres) == 0:
            return []
        vec = list()
        for pre in pres:
            vec.append(self.vectorize_sen(pre))
            vec.append(self.token2id[SEP])
        return vec[:-1]


    def pred2vec(self):
        pass

    def recover(self,seq):
        return  [self.id2token.get(s) for s in seq]




if __name__ == '__main__':
    with open('../model/phrase_rank.json', 'r', encoding='utf-8') as fp:
        phrase_rank = json.load(fp)
    with open('../model/token2id.json', 'r', encoding='utf-8') as fp:
        token2id = json.load(fp)
    dg = DatasetGen(token2id, phrase_rank)
    t,v,tt = dg.gen_train_val_test([
        ['寒惊蓟门叶E', '秋发小山枝E', '松阴背日转E', '竹影避风移E'],
        ['寒随穷律变E', '春逐鸟声开E' , '初风飘带柳E', '晚雪间花梅E'],
        ['萧条起关塞E', '摇飏下蓬瀛E', '拂林花乱彩E', '响谷鸟分声E', '披云罗影散E', '泛水织文生E']
          ])
    trace = t+v+tt
    print(len(trace))
    for triple in trace:
        keyword = dg.recover(triple[0])
        preceeding_text = dg.recover(triple[1])
        current_line = dg.recover(triple[2])
        print(keyword)
        print(preceeding_text)
        print(current_line)
        print('===============')
