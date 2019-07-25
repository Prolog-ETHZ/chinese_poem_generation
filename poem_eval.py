import argparse
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from processing.dataset import PoemDataset
from processing.constants import *
import math


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
    parser.add_argument("--pre_trained_model", type=str, default="./model/trained_models/poem_model")
    args = parser.parse_args()
    return args

def test(opts):

    with open('./model/id2token.json', 'r', encoding='utf-8') as fp:
        id2token = json.load(fp)
    with open('./model/token2id.json', 'r', encoding='utf-8') as fp:
        token2id = json.load(fp)

    # set random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_params = {"batch_size": opts.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if torch.cuda.is_available():
        poem_model = torch.load(opts.pre_trained_model)
    else:
        poem_model = torch.load(opts.pre_trained_model, map_location=lambda storage, loc: storage)

    test_set = PoemDataset(opts.test_set)
    test_generator = DataLoader(test_set, **test_params)
    poem_model.to(device)
    poem_model.eval()
    epoch_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=int(token2id[PAD]))

    with torch.no_grad():
        for i, (keywords, preceding_texts, current_lines) in enumerate(test_generator):
            # print(keywords)
            # print(type(keywords))
            # exit(0)
            keywords = keywords.transpose(0, 1).to(device)
            preceding_texts = preceding_texts.transpose(0, 1).to(device)
            current_lines = current_lines.transpose(0, 1).to(device)
            # turn off teacher forcing
            output = poem_model(keywords, preceding_texts, current_lines, 0)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            output = output[1:].view(-1, output.shape[-1])
            current_lines = current_lines[1:].contiguous().view(-1)
            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, current_lines)
            epoch_loss += loss.item()
            # print(i)

    test_loss = epoch_loss / len(test_generator)
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. PPL: {math.exp(test_loss):7.3f}')






if __name__ == "__main__":
    # specify the arguments first
    opts = get_args()
    test(opts)
