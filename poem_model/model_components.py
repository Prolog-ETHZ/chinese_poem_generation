import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, enc_layer,
                 dec_hid_dim, dropout, pretrained_embed=None):
        """

        :param vocab_size: the vocab size
        :param emb_dim:  the embedding dim
        :param enc_hid_dim:  the encoder hidden size
        :param dec_hid_dim:  the decoder hidden size
        :param dropout: dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embed is not None:
            self.embedding.load_state_dict({'weight': torch.from_numpy(pretrained_embed)})
            self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=enc_layer)
        # TODO: might be add this linear transform from enc to dec later
        # self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        # p: probability of an element to be zeroed. Default: 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], hidden.shape[1]*hidden.shape[2])
        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, enc hid dim * n layers * num directions]
        return outputs, hidden


class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        """

        :param enc_hid_dim: dim of encoder
        :param dec_hid_dim: dim of decoder
        """
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        # TODO: check if this is vars because v here is actually acts as the context vector
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim] is h_{t-1} of decoder
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # return the softmaxed attention score for each e_{0...T}
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        # unsqueeze add a extra dimension with 1
        # [batch size, dec hid dim] -> [batch size, 1,dec hid dim]
        # repeat function repeat the original tensor multiple times along each dim
        # [batch size, 1,dec hid dim] -> [batch size, src_len, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        # after cat, [batch size, src sent len, enc hid dim * 2+dec hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]
        # bmm Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        # b * n *m  bmm b*m*p => b*n*p
        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim,
                 dec_layer, dropout, attention, pretrained_embed=None):

        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if pretrained_embed is not None:

            self.embedding.load_state_dict({'weight': torch.from_numpy(pretrained_embed)})
            self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=dec_layer)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size] -> y_{t-1}, the actual word index
        # hidden = [batch size, dec hid dim] -> h_{t-1}
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2] -> e_{0..T}

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2 * num_layer]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2] -> c_{i}

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]
        # rnn_input = [x_{t};h_{t-1}]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # TODO: hidden shape is wrong
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        # self.rnn(h_{t-1}, c_{i}, y_{i-1})
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        # output is h_{t}, weight is c_{t}=h_{t-1} embed is x_{t} = y_{t-1}
        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0)


class Poem2Seq(nn.Module):

    def __init__(self, vocab_size, enocder_dim, decoder_dim, embed_dim,
                 encoder_layer, decoder_layer,
                 device, dropout_prob=0.7,
                 pretrained_embed=None):

        """

        :param vocab_size: the vocab size for the task
        :param enocder_dim: hidden state size of encoder
        :param decoder_dim: hidden state size of decoder
        :param encoder_layer: layer of RNN
        :param decoder_layer: layer of RNN
        :param embed_dim: word embedding dim
        :param dropout_prob: dropout rate
        :param pretrained_embed: pretrained word embedding table

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = enocder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.pretrained_embed = pretrained_embed
        # keyword is only one or two words, so the layer is always be 1
        self.keyword_encoder = Encoder(vocab_size, embed_dim, enocder_dim, 1,decoder_dim,
                                       dropout_prob, pretrained_embed)

        self.text_encoder = Encoder(vocab_size, embed_dim, enocder_dim, encoder_layer,
                                    decoder_dim, dropout_prob, pretrained_embed)

        self.decoder =  Decoder(vocab_size, embed_dim, enocder_dim, decoder_dim,
                                decoder_layer,
                                dropout_prob,
                                Attention(enocder_dim, decoder_dim), pretrained_embed)
        self.device = device
        self.fc = nn.Linear(self.encoder_dim * 2 * encoder_layer , self.decoder_dim)


    def forward(self, keyword, source_text, target_text, 
        teacher_forcing_ratio=0.9, beam_search_mode = None, allow_repeat=True):

        # keyword = [keyword sent len, batch size] -> 4
        # source_text = [source_text, batch size] -> 50
        # target_text = [target_text, batch size] -> 8
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = source_text.shape[1]
        max_len = target_text.shape[0]
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, enc hid dim * 2]
        text_encoder_outputs, text_hidden = self.text_encoder(source_text)
        keyword_encoder_outputs, keyword_hidden = self.keyword_encoder(keyword)
        encoder_outputs = torch.cat((keyword_encoder_outputs, text_encoder_outputs), dim=0)

        # hidden = [batch size, decoder dim]
        hidden = torch.tanh(self.fc(text_hidden))

        if not beam_search_mode:
            # first input to the decoder is the BEGIN_TOKEN tokens
            output = target_text[0, :]
            trace = [output]
            # outputs[0] = output
            # TODO: decoder out is without softmax
            # print(output)
            for t in range(1, max_len):
                # hidden is h_{t-1}, initially h_{0} the output of text encoder
                # print(output.shape)
                # print(hidden.shape)
                # print(encoder_outputs.shape)
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                # output stores all P(w|network)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                # tensor.max return a pair as (values, indices)
                top1 = output.max(1)[1]
                trace.append(top1)
                # top1 is the index of the word
                output = (target_text[t] if teacher_force else top1)

            return outputs

        else:

            output = target_text[0,:]            
            # hidden is h_{t-1}, initially h_{0} the output of text encoder
            # encoder output is E_{0..T}, output is y_{t-1}
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            # output is (batch_size , vocab_size)
            # the first fill
            output = F.softmax(output, dim=1)
            values, indices = torch.topk(output, beam_search_mode, dim=1)
            indices = torch.flatten(indices)
              
            prev_beam_trace = np.zeros((beam_search_mode, max_len), dtype=np.int64)
            prev_beam_trace[:, 0] = target_text[0,:].cpu().numpy()
            prev_beam_trace[:, 1] = torch.flatten(indices).cpu().numpy()
            prev_probs = values.permute(1,0)

            encoder_outputs = encoder_outputs.repeat(1, beam_search_mode, 1)
            prev_beam_outputs = indices
            prev_beam_hidden = hidden.repeat(beam_search_mode, 1)

            # print(prev_beam_trace)
            # print(prev_probs)
            # print(prev_beam_outputs.shape)
            # print(prev_beam_hidden.shape)
            # print(encoder_outputs.shape)
            

            for t in range(2, max_len):
                # hidden is h_{t-1}, initially h_{0} the output of text encoder
                # print(prev_beam_hidden.type())
                output, hidden = self.decoder(prev_beam_outputs, prev_beam_hidden, encoder_outputs)
                output = F.softmax(output, dim=1)
                output = output * prev_probs
                values, indices = torch.topk(torch.flatten(output), beam_search_mode*4)
                
                beam_trace = np.zeros((beam_search_mode, max_len))
                beam_outputs = np.ones(beam_search_mode, dtype=np.int64)
                beam_hidden = torch.ones(hidden.shape, dtype=torch.float32, device=self.device)
                probs = torch.ones(beam_search_mode, dtype=torch.float32, device=self.device)
                slot = 0

                for i, index in enumerate(indices.cpu().numpy()):

                    pos = int(index / self.vocab_size)
                    real_index = index % self.vocab_size
                    # print(real_index, prev_beam_trace[slot][:t])
                    if real_index in set(prev_beam_trace[pos]):
                        # print('??')
                        continue
                    beam_trace[slot] = prev_beam_trace[pos]
                    beam_trace[slot][t] = real_index
                    beam_outputs[slot] = real_index
                    beam_hidden[slot] = hidden[pos]
                    probs[slot] = values[pos]

                    slot += 1
                    if slot == 10:
                        break

                
                prev_beam_outputs = torch.from_numpy(beam_outputs).to(self.device)
                prev_beam_hidden = beam_hidden
                prev_beam_trace = beam_trace
                prev_probs = probs.unsqueeze(1)
                # prev_probs = values.unsqueeze(1)

            # print(prev_beam_trace)
            # print(prev_probs)
            # print(prev_beam_outputs.shape)
            # print(prev_beam_hidden.shape)
            # print(encoder_outputs.shape)
            # exit(0)
            return prev_beam_trace[0]







        """
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(source_text)

        # first input to the decoder is the <sos> tokens
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
        """



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Poem2Seq(vocab_size=10, enocder_dim=32, decoder_dim=32, embed_dim=64, device=device,
                     dropout_prob=0.7,
                 pretrained_embed=None).to(device)
    print(model.apply(init_weights))
    print(f'The model has {count_parameters(model):,} trainable parameters')
    for param in model.parameters():
        print(type(param.data), param.size())
