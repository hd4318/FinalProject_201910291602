# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import torch.nn as nn # Nueral Network에 대한 package
from module.rnn_base import EncoderRNN, LuongDecoder

class Seq2Seq(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.SOS = config.get("start_index", 1)  # Start index를 가져옵니다.
        self.batch_size = config.get("batch_size", 1)  # batch_size 정보를 가져옵니다.
        self.gpu = config.get("gpu", False)  # cuda 로 돌아가는지 아닌지에 대한 정보
        self.vocab_size = config["tgt_vocab_size"]

        # Encoder 선언

        self.encoder = EncoderRNN(config)

        # Decoder 선언

        self.decoder = LuongDecoder(config)

        # loss fucntion
        # ignore_index =0 왜???
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    def encode(self, x, x_len):
        # encoder를 통해 주어진 source 정보를 Encodeing 하는 용도

        batch_size = x.size()[0]
        # 초기 inital hidden state 만들기
        init_state = self.encoder.init_hidden(batch_size, self.config)
        # encoder Forward 수행
        encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)

        return encoder_outputs, encoder_state

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """

        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        # decoder의 처음 y0 는 무엇이 되어야 할까? *주의해야할 포인트
        if batch_size == 1:
            decoder_input = torch.LongTensor([self.SOS] * batch_size)
        else:
            decoder_input = torch.LongTensor([self.SOS] * batch_size).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]  # (Batch,1)
        decoder_hidden = encoder_hidden
        # alignments :  attention align을 저장하기 위한 용도
        alignments = torch.zeros(max_length, encoder_outputs.size(1), batch_size)  # attention align을 저장하기 위한 용도
        logits = torch.zeros(max_length, batch_size, self.decoder.output_size)  # logits 값을 저장하기 위한 용도의 tensor

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()
        inference = []
        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]

            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            outputs, decoder_hidden, attention_weights = self.decoder.forward(
                input=decoder_input.long(),
                last_hidden=decoder_hidden,
                Encoder_Outputs=encoder_outputs,
                Seq_Len=input_lengths)

            alignments[t] = attention_weights.transpose(1, 0)
            #alignments[t] = torch.zeros((attention_weights.shape[1], attention_weights.shape[0]))

            logits[t] = outputs

            if self.training:
                decoder_input = targets[:, t]
            else:
                topv, topi = outputs.data.topk(1)  # 가장 높은 예측만 사용.
                decoder_input = topi.squeeze(-1).detach()
                inference.append(decoder_input.cpu())

        labels = targets.contiguous().view(-1)

        mask_value = 0
        # what is this mask_3d? # (warning check)
        #print("A0", logits)
        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)
        #print("A1", logits)
        logits = logits.contiguous().view(-1, self.vocab_size)  # loss를 구하기 위해 쫙 펴주기
        #print("A2", logits)

        return logits, labels.long(), alignments, inference



    def step(self, batch):
        x, x_len, y, y_len = batch
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
            x_len = x_len.cuda()
            y_len = y_len.cuda()

        encoder_out, encoder_state = self.encode(x, x_len)  # encoder
        logits, labels, alignments, inference = self.decode(encoder_out, encoder_state, y, y_len,
                                                            x_len)  # decoder 를 통해 alignment와 logit 값 얻기
        return logits, labels, alignments, inference

    def loss(self, batch):
        logits, labels, alignments, inference = self.step(batch)
        loss = self.loss_fn(logits, labels)  # loss 구하기 우리는 cross entropy 사용
        return loss, logits, labels, alignments, inference

## 추후에 설명 Decoder section
def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len) # length 체크
    max_idx = max(seq_len) # max length 체크
    for n, idx in enumerate(seq_len): # length 에서 의미없는 hidden state attention 값은 0으로 두기 위한 mask값 설정
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs