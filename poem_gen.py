import argparse
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from processing.dataset import PoemDataset
from processing.constants import *
import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def get_args():
    parser = argparse.ArgumentParser(
        """Hyper parameters for poem model""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=5)
    parser.add_argument("--enocder_dim", type=int, default=64)
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--check_step", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=20)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--test_set", type=str, default="./model/test.pkl")
    parser.add_argument("--val_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word_embed_path", type=str, default="./model/embed_table.npy")
    parser.add_argument("--pre_trained_model", type=str, default="./model/trained_models/poem_model_0")
    args = parser.parse_args()
    return args

def gen_poems(opts, keywords, beam_mode):

	with open('./model/id2token.json', 'r', encoding='utf-8') as fp:
	    id2token = json.load(fp)
	with open('./model/token2id.json', 'r', encoding='utf-8') as fp:
	    token2id = json.load(fp)
	#print(id2token[str(2522)])
	# set random seed
	if torch.cuda.is_available():
	    torch.cuda.manual_seed(123)
	else:
	    torch.manual_seed(123)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if torch.cuda.is_available():
	    model = torch.load(opts.pre_trained_model)
	else:
	    model = torch.load(opts.pre_trained_model, map_location=lambda storage, loc: storage)
	model = model.to(device)
	model.eval()

	keywords = [[token2id[c] if c in token2id else token2id[UNK] for c in k ] for k in keywords]
	preceding_texts = [[token2id[BEGIN_TOKEN], token2id[END_TOKEN]]]
	current_lines = [[token2id[BEGIN_TOKEN]]]
	# print(keywords)
	# X
	keywords = np.asarray(pad_sequences(keywords, maxlen=5,
	                                    truncating='pre',
	                                    padding='pre'
	                                    ))

	preceding_texts = np.asarray(pad_sequences(preceding_texts, maxlen=25,
	                                            truncating='pre',
	                                            padding='pre'
	                                            ))
	# Y
	current_lines = np.asarray(pad_sequences(current_lines, maxlen=9,
	                                         truncating='post',
	                                         padding='post'
	                                         ))

	keywords = torch.from_numpy(keywords).long()
	preceding_texts = torch.from_numpy(preceding_texts).long()
	current_lines = torch.from_numpy(current_lines).long()
	# print(keywords)
	# print(preceding_texts)
	# print(current_lines)
	# gen_func = gen_sentence_beam if beam_mode else gen_sentence
	for i in range(4):
		last = (i == 3)
		input_keyword = keywords[i]
		input_keyword = input_keyword.unsqueeze(0)
		# print(input_keyword.shape)
		if beam_mode:
		    preceding_texts = gen_sentence_beam(model, device, input_keyword,
		    	preceding_texts, current_lines,id2token, token2id, beam_mode, last=last)
		else:
			preceding_texts = gen_sentence(model, device, input_keyword,
				preceding_texts, current_lines,id2token, token2id, last=last)

	print_poems(preceding_texts[0],id2token)


def print_poems(preceding_texts ,id2token):
	# print(len(preceding_texts))
	text = [id2token[str(p)] for p in preceding_texts.cpu().numpy()]
	line = ""
	for t in text:
		if t == PAD:
			continue
		elif t == SEP or t == END_TOKEN:
			print(line)
			line = ""

		else:
			line += t



def gen_sentence_beam(model, device, keywords, preceding_texts, current_lines,
                 id2token, token2id, beam_size=10, last=False):

	with torch.no_grad():

		keywords = keywords.transpose(0, 1).to(device)
		preceding_texts = preceding_texts.transpose(0, 1).to(device)
		current_lines = current_lines.transpose(0, 1).to(device)
		# NOTE: the output of beams search mode will be directly the index rather than the probs values
		indices = model(keywords, preceding_texts, current_lines, 0, beam_search_mode = beam_size)
		line = [id2token[str(int(i))] for i in indices]
		if line[0] == BEGIN_TOKEN:
		    del line[0]
		if line[-1] == END_TOKEN:
		    del line[-1]
		# print(line)
		preceding = [id2token[str(i.cpu().numpy())] for i in preceding_texts.squeeze(1)
		             if id2token[str(i.cpu().numpy())]
		             not in {BEGIN_TOKEN, END_TOKEN, PAD}
		             ]

		if len(preceding) == 0:
		    preceding = [token2id[BEGIN_TOKEN]]+[token2id[l] for l in line]+[token2id[END_TOKEN]]
		else:
		    preceding = [token2id[p] for p in preceding]+[token2id[SEP]]+[token2id[l]
		                                                                  for l in line]+[token2id[END_TOKEN]]

		# print([id2token[str(p)] for p in preceding])
		maxlen = 25 if not last else 50
		preceding_texts = np.asarray(pad_sequences([preceding], maxlen=maxlen,
		                                           truncating='post',
		                                           ))
		preceding_texts = torch.from_numpy(preceding_texts).long()
		return preceding_texts

def gen_sentence(model, device, keywords, preceding_texts, current_lines,
                 id2token, token2id, last=False):

    with torch.no_grad():

        keywords = keywords.transpose(0, 1).to(device)
        preceding_texts = preceding_texts.transpose(0, 1).to(device)
        current_lines = current_lines.transpose(0, 1).to(device)
        output = model(keywords, preceding_texts, current_lines, 0)
        output = output.squeeze(1)[1:]
        values, indices = torch.max(output, 1)
        line = [id2token[str(i.cpu().numpy())] for i in indices]
        if line[0] == BEGIN_TOKEN:
            del line[0]
        if line[-1] == END_TOKEN:
            del line[-1]
        # print(line)
        preceding = [id2token[str(i.cpu().numpy())] for i in preceding_texts.squeeze(1)
                     if id2token[str(i.cpu().numpy())]
                     not in {BEGIN_TOKEN, END_TOKEN, PAD}
                     ]

        if len(preceding) == 0:
            preceding = [token2id[BEGIN_TOKEN]]+[token2id[l] for l in line]+[token2id[END_TOKEN]]
        else:
            preceding = [token2id[p] for p in preceding]+[token2id[SEP]]+[token2id[l]
                                                                          for l in line]+[token2id[END_TOKEN]]

        # print([id2token[str(p)] for p in preceding])
        maxlen = 25 if not last else 50
        preceding_texts = np.asarray(pad_sequences([preceding], maxlen=maxlen,
                                                   truncating='post',
                                                   ))
        preceding_texts = torch.from_numpy(preceding_texts).long()
        return preceding_texts



if __name__ == "__main__":
    # specify the arguments first
    opts = get_args()
    print('Generate Poems with beam search.....')
    
    print('<春天>')
    gen_poems(opts, ['春天','桃花', '燕', '柳'], beam_mode=10)
    print('<思乡>')
    gen_poems(opts, ['明月','酒', '亲人', '思乡'], beam_mode=10)
    print('<秋天>')
    gen_poems(opts, ['秋','落叶', '雨', '孤灯'], beam_mode=10)
    print('<登山>')
    gen_poems(opts, ['高山','流水', '古琴', '君子'], beam_mode=10)
    """
    print('Generate Poems without beam search.....')
    
    print('<春天>')
    gen_poems(opts, ['春天','桃花', '燕', '柳'], beam_mode=None)
    print('<思乡>')
    gen_poems(opts, ['明月','酒', '亲人', '思乡'], beam_mode=None)
    print('<秋天>')
    gen_poems(opts, ['秋','落叶', '雨', '孤灯'], beam_mode=None)
    print('<登山>')
    gen_poems(opts, ['高山','流水', '古琴', '君子'], beam_mode=None)
    """






