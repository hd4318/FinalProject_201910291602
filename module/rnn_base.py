# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F # pytorch function 들을 사용하기 위한 용도

from util.tokens import PAD_TOKEN_ID
from util.tokens import SOS_TOKEN_ID


class GruEncoder(nn.Module):
    """Gru Encoder"""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.device = kwargs.get('device', 'cpu')

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=PAD_TOKEN_ID)
        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          batch_first=True,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          dropout=self.dropout_prob)

    def forward(self, x, seq_lengths):
        embedding = self.embedding_lookup(x)
        packed_input = pack_padded_sequence(embedding, seq_lengths, batch_first=True)

        # If bidirectional is True,
        # output shape : (batch_size, seq_len, 2 * hidden_size)
        # hidden shape : (2 * num_layers, batch_size, hidden_size)
        output, hidden_state = self.rnn(packed_input)

        # output shape : (batch_size, seq_len, 2 * hidden_size)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=PAD_TOKEN_ID)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            hidden_state = hidden_state[:self.num_layers] + hidden_state[self.num_layers:]

        # Standard rnn decoder cannot be bidirectional...
        return output, hidden_state

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)


class GruDecoder(nn.Module):
    """Gru Decoder"""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.device = kwargs.get('device', 'cpu')

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=PAD_TOKEN_ID)
        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          batch_first=True,
                          bidirectional=False,
                          num_layers=self.num_layers,
                          dropout=self.dropout_prob)
        self.linear_transform = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder_output_func = nn.functional.log_softmax

    def forward(self, encoder_output, encoder_hidden_state, tgt_seqs, tgt_seq_lengths):
        # Decoder GRU cannot be bidirectional.
        # encoder output:  (batch, seq_len, num_directions * hidden_size) => batch_first=True
        # encoder hidden:  (num_layers * num_directions, batch, hidden_size)

        batch_size = encoder_output.size(0)
        max_seq_len = tgt_seqs.size(-1)
        # (Batch_size)
        initial_input = batch_size * [SOS_TOKEN_ID]
        initial_input = torch.tensor(initial_input, dtype=torch.long, device=self.device).unsqueeze(
            -1)

        # predicted output will be saved here
        logits = torch.zeros(max_seq_len, batch_size, self.vocab_size, device=self.device)

        decoder_input = initial_input
        prev_hidden_state = encoder_hidden_state

        predictions = []
        for t in range(tgt_seqs.size(-1)):
            decoder_output, hidden_state = self.step(t, decoder_input, prev_hidden_state)
            logits[t] = decoder_output

            if self.training:
                decoder_input = tgt_seqs[:, t]
            else:
                # Greedy search
                top_value, top_index = decoder_output.data.topk(1)
                decoder_input = top_index.squeeze(-1).detach()
                predictions.append(decoder_input.cpu())

            decoder_input = decoder_input.long().unsqueeze(-1)
            prev_hidden_state = hidden_state

        # To calculate loss, we should change shape of logits and labels
        # N is batch * seq_len, C is number of classes. (vocab size)
        # logits : (N by C)
        # labels : (N)
        logits = logits.transpose(0, 1)
        logits = logits.contiguous().view(-1, self.vocab_size)
        labels = tgt_seqs.contiguous().view(-1)

        return logits, labels, predictions

    def step(self, t, inputs, prev_hidden_state):
        embedding = self.embedding_lookup(inputs)

        outputs, hidden_state = self.rnn(embedding, prev_hidden_state)
        outputs = self.linear_transform(outputs.transpose(0, 1).squeeze(0))

        if self.decoder_output_func:
            outputs = self.decoder_output_func(outputs, dim=-1)

        # To save in logits, seq_len should be removed.
        return outputs, hidden_state

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.layers = config.get("encoder_layers", 1)
        self.vocab_size = config["src_vocab_size"]
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        gru_input_dim = self.embedding_dim
        self.rnn = nn.GRU(
            gru_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)  # model 선언
        self.gpu = config.get("gpu", False)

    def forward(self, inputs, hidden, input_lengths):
        ## (To do) 이 부분의 코드를 완성하시오!
        # 제공된 코드를 수정하지 않고 forward 문만 작성하여 코드를 구현해 주세요!
        # 기존 코드를 수정하지 않고 코드를 구현해 주세요!
        inputs = self.embedding(inputs)

        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, state = self.rnn(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

        if self.bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            state = state[:self.layers] + state[self.layers:]
        return output, state

    def init_hidden(self, batch_size, config):
        ## (To do) 이 부분의 코드를 완성하시오!
        # 제공된 코드의 다른 부분을 수정하지 않고 init_hidden 코드를 구현해 주세요!
        # 기존 코드를 수정하지 않고 코드를 구현해 주세요!
        h0 = torch.zeros(2 * self.layers if self.bi else self.layers, batch_size, self.hidden_size)
        if self.gpu:
            h0 = h0.cuda()
        return h0


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """

    def __init__(self, batch_size, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            # Wa (hidden,hidden)
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            # Wa : (2*hidden,hidden)
            # Va : (hidden,1)
            self.Wa = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        elif method == 'bahdanau':
            # Wa : (hidden_size,hidden_size)
            # Ua : (hidden_size,hidden_size)
            # Va : (hidden_size,1)
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        else:
            raise NotImplementedError

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        """
        Inputs :
          last_hidden : (B,T,hidden_size)
          encoder_outputs :
          seq_len:
        Returns:
          attention matrix :
        """
        batch_size, seq_lens, _ = encoder_outputs.size()
        # attention energies 를 구하기
        attention_energies = self.score(last_hidden, encoder_outputs, self.method)

        if seq_len is not None:
            attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def score(self, last_hidden, encoder_outputs, method):
        # (To do) 코드를 완성하시오
        # 기존 코드를 수정하지 않고 코드를 구현해 주세요!
        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)  # (batch_size, hidden_dim,1)

            # attention : (batch_size,max_time, hidden_dim) , (batch_size,hidden_dim,1) - > (batch_size,max_time ,1)

            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            # dot 이랑 비슷 다만 last hidden을 한번 projection
            x = self.Wa(last_hidden)  # (batch_size, hidden_dim) ->  (batch_size, hidden_dim)
            x = x.unsqueeze(-1)  # (batch_size, hidden_dim) ->  (batch_size, hidden_dim,1)
            # encoded 된 hidden states 와 dot proudct를 수행하기
            # attention: (batch_size,max_time, hidden_dim) , (batch_size,hidden_dim,1) - > (batch_size,max_time ,1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1).expand_as(
                encoder_outputs)  # (batch_size, hidden_dim) ->  (batch_size,1, hidden_dim)
            # concat 후 -> linear 거치기 -> 후 tanh
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs),
                                         -1)))  # (batch_size, max_timestep, hidden_dim) ->  (batch_size,  max_timestep, hidden_dim*2)
            # (batch_size, max_timestep, hidden_dim*2) ->  (batch_size,  max_timestep, )
            return x.matmul(self.va).squeeze(-1)

        elif method == "bahdanau":
            # mlp 기반의 attention model

            x = last_hidden.unsqueeze(1)  # (batch_size, hidden_dim) ->  (batch_size,1, hidden_dim)
            # 각각을 projection 후 더하기 -> tanh
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))  #
            return out.matmul(self.va).squeeze(
                -1)  # (batch_size,max_timestep,hidden_dim) ->  (batch_size, max_timestep)

        elif method == "luong":
            # mlp 기반의 attention model

            x = last_hidden.unsqueeze(1)  # (batch_size, hidden_dim) ->  (batch_size,1, hidden_dim)
            # 각각을 projection 후 더하기 -> tanh
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))  #
            return out.matmul(self.va).squeeze(
                -1)  # (batch_size,max_timestep,hidden_dim) ->  (batch_size, max_timestep)

        else:
            raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.batch_size = config["batch_size"]
        self.hidden_size = config["decoder_hidden"]
        self.vocab_size = config["tgt_vocab_size"]
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.Embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.GRU = nn.GRU(
            input_size=self.embedding_dim+self.hidden_size if config['decoder'].lower() == 'bahdanau' else self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=config.get("decoder_layers", 1),
            dropout=config.get("decoder_dropout", 0),
            bidirectional=False,
            batch_first=True)
        if config['decoder'] != "RNN":
            self.Attention = Attention(
                self.batch_size,
                self.hidden_size,
                method=config.get("attention_score", "dot"))

        self.gpu = config.get("gpu", False)
        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError


class LuongDecoder(Decoder):
    """
        Corresponds to LoungAttnDecoderRNN
    """

    def __init__(self, config):
        super(LuongDecoder, self).__init__(config)
        self.output_size = config["tgt_vocab_size"]
        self.outputs2vocab = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, **kwargs):

        # 기존 코드를 수정하지 않고 코드를 구현해 주세요!
        input = kwargs["input"]  # decoder input
        prev_hidden = kwargs["last_hidden"]  # decoder rnn 에서 들어갈 previous hidden state
        encoder_outputs = kwargs["Encoder_Outputs"]  # encoder RNN에서 Encoding이 끝난 (B,L,hidden_size)
        seq_len = kwargs.get("seq_len", None)  # sequence length

        # check inputs

        # To do start

        # Attention weights

        # embed characters
        embedded = self.Embedding(input)

        rnn_input = embedded
        # runn rnn
        outputs, hidden = self.GRU(rnn_input.unsqueeze(1), prev_hidden)  # 1 x B x N, B x N

        # weights get
        weights = self.Attention.forward(outputs.squeeze(1), encoder_outputs, seq_len)  # B x T
        # get context vector
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x H]
        # concat context vector and rnns' output
        output_cat = torch.cat((outputs.squeeze(1), context), 1)
        # predict next tokens
        output = self.outputs2vocab(output_cat)  # logit 값 각 chracter 별로

        # to do end this points
        if self.decoder_output_fn:
            # NLL loss 인 경우
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden, weights