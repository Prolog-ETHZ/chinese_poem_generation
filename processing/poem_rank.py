import networkx as nx
from networkx import exception
import jieba
from tqdm import *

class PoemTextRank(object):

    def __init__(self):

        self.graph = nx.Graph()  # initialize a undirected graph

    def add_edge(self, v1, v2):

        if not self.graph.has_node(v1):
            self.graph.add_node(v1)

        if not self.graph.has_node(v2):
            self.graph.add_node(v2)

        if not self.graph.has_edge(v1, v2):
            self.graph.add_edge(v1, v2, weight=1)

        else:
            self.graph[v1][v2]['weight'] += 1

    def compute_page_rank(self):

        calculated_page_rank = None
        try:
            calculated_page_rank = nx.pagerank(self.graph, weight='weight')
        except exception.PowerIterationFailedConvergence:
            for u, v, a in self.graph.edges(data=True):
                print(u, v, a)
            exit(0)

        return calculated_page_rank


"""
compute on all poems and context window is one poem (or one poem sen)
"""
def compute_poem_rank(poems, bpe_encoder):
    ptr = PoemTextRank()
    for poem in tqdm(poems):
        '''
        for sen in poem:
            seg_list = jieba.cut(sen[:-1], cut_all=False)
            poem_phrases.extend(seg_list)

        # print(poem_phrases)
        # input('continue...')
        '''
        # for sen in poem:
            # print(sen[:-1])
        #    seg_list = bpe_encoder.tokenize(sen[:-1])[1:-1]
        #    poem_phrases.extend(seg_list)
        # print(poem_phrases)
        # input('continue...')
        # print(poem_phrases)

        # character level text rank score

        for sen in poem:
            poem_phrases = sen[:-1]
            for i in range(len(poem_phrases)):
                for j in range(i):
                    ptr.add_edge(poem_phrases[i], poem_phrases[j])

    # print(len(ptr.graph))
    # exit(0)
    return ptr.compute_page_rank()
