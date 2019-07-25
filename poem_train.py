import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from processing.dataset import PoemDataset
from poem_model.model_components import Poem2Seq
import numpy as np
from processing.constants import *
from torch import optim
import math
import os
import json

def get_args():
    parser = argparse.ArgumentParser(
        """Hyper parameters for poem model""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--enocder_dim", type=int, default=256)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--encoder_layer", type=int, default=2)
    parser.add_argument("--decoder_layer", type=int, default=1)

    parser.add_argument("--check_step", type=int, default=200)
    # parser.add_argument("--lr", type=float, default=0.01) #
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--teacher_force", type=float, default=0.8)
    parser.add_argument("--train_set", type=str, default="./model/train.pkl")
    parser.add_argument("--val_set", type=str, default="./model/val.pkl")
    parser.add_argument("--val_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word_embed_path", type=str, default="./model/embed_table.npy")
    parser.add_argument("--saved_path", type=str, default="./model/trained_models")
    args = parser.parse_args()
    return args

def train(opts):
    with open('./model/id2token.json', 'r', encoding='utf-8') as fp:
        id2token = json.load(fp)
    with open('./model/token2id.json', 'r', encoding='utf-8') as fp:
        token2id = json.load(fp)
    # set random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # load the dataset
    train_params = {"batch_size": opts.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    val_params = {"batch_size": opts.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    training_set = PoemDataset(opts.train_set)
    training_generator = DataLoader(training_set, **train_params)
    val_set = PoemDataset(opts.val_set)
    val_generator = DataLoader(val_set, **val_params)

    # load word embedding
    pretrained_embed = np.load(opts.word_embed_path)
    print(pretrained_embed.shape)
    vocab_size = pretrained_embed.shape[0]
    embed_dim = pretrained_embed.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init the model
    """
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Poem2Seq(vocab_size=10, enocder_dim=32, decoder_dim=32, embed_dim=64, device=device,
                     dropout_prob=0.7,
                 pretrained_embed=None).to(device)
    """
    poem_model = Poem2Seq(vocab_size, opts.enocder_dim, opts.decoder_dim,
                          embed_dim, opts.encoder_layer, opts.decoder_layer,
                          device, dropout_prob=opts.dropout_prob,
                          pretrained_embed=pretrained_embed)

    poem_model = poem_model.to(device)

    # set up the training configurations
    criterion = nn.CrossEntropyLoss(ignore_index = int(token2id[PAD]))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, poem_model.parameters()))
    best_valid_loss = 1e5
    best_epoch = 0


    # start training
    for epoch in range(opts.num_epoches):

        poem_model.train()
        total_loss = 0.0
        # iter is the batch number, feature is training feats and labels
        for iter, (keywords, preceding_texts, current_lines) in enumerate(training_generator):

            # TODO: make sure shape order is correct
            # actual training steps

            keywords = keywords.transpose(0,1).to(device)
            preceding_texts = preceding_texts.transpose(0,1).to(device)
            current_lines = current_lines.transpose(0,1).to(device)

            # case = keywords[:,0]
            # print([id2token[str(int(c.numpy()))] for c in case])
            # print(keywords)
            # exit(0)
            # print(keywords.shape)
            # print(current_lines.shape)
            # print(preceding_texts.shape)
            # exit(0)
            # TODO: check from here
            optimizer.zero_grad()
            output = poem_model(keywords, preceding_texts, current_lines,
                                teacher_forcing_ratio=opts.teacher_force)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            output = output[1:].view(-1, output.shape[-1])
            current_lines = current_lines[1:].contiguous().view(-1)
            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, current_lines)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(poem_model.parameters(), opts.clip)
            optimizer.step()
            # print(iter)
            # dynamic monitoring of the performance
            total_loss += loss.data
            if iter % opts.check_step == 0 and iter != 0:
                total_loss = total_loss / opts.check_step
                print("[%d][loss:%5.2f][pp:%5.2f]" % (iter, total_loss, math.exp(total_loss)))
                total_loss = 0.0

        # check on validation set
        if (epoch+1) % opts.val_interval == 0:
            print('begin validation......')
            valid_loss = validate(poem_model, val_generator, criterion, device)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
            
            torch.save(poem_model, opts.saved_path + os.sep + "poem_model_"+str(epoch))
            print(f'Epoch: {epoch+1:02}, current best epoch: {best_epoch:02}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def validate(poem_model, val_generator, criterion, device):

    poem_model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (keywords, preceding_texts, current_lines) in enumerate(val_generator):

            keywords = keywords.transpose(0, 1).to(device)
            preceding_texts = preceding_texts.transpose(0, 1).to(device)
            current_lines = current_lines.transpose(0, 1).to(device)
            # turn off teacher forcing
            output = poem_model(keywords, preceding_texts, current_lines, 0.0)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            # TODO: 1: because pre-zero padding
            output = output[1:].view(-1, output.shape[-1])
            current_lines = current_lines[1:].contiguous().view(-1)
            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, current_lines)
            epoch_loss += loss.item()

    return epoch_loss / len(val_generator)


def main():
    # specify the arguments first
    opts = get_args()
    train(opts)

if __name__ == "__main__":
    # specify the arguments first
    opts = get_args()
    train(opts)



