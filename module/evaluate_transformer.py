from tqdm import tqdm
import torch
import nltk
import numpy as np
from chatspace import ChatSpace

spacer = ChatSpace()

def correct_sum(y_pred, dec_output):
    with torch.no_grad():
        y_pred = y_pred.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
        correct_elms = (y_pred == dec_output).float()[dec_output != 0]
        correct_sum = correct_elms.sum().to(torch.device('cpu')).numpy()  # padding은 acc에서 제거
        num_correct_elms = len(correct_elms)
    return correct_sum, num_correct_elms


class token_decoder:

    # def __init__(self, input_ids, pred_ids, src_id2word, tgt_id2word,
    #              src_seqs, y_pred, dec_output=None):
    #     self.input_ids = input_ids
    #     self.pred_ids = pred_ids
    #     self.src_id2word = src_id2word
    #     self.tgt_id2word = tgt_id2word
    #     self.src_seqs = src_seqs
    #     self.y_pred = y_pred
    #     self.dec_output = dec_output

    def decode_src_token_ids(self, input_ids, src_id2word):
        token_token_batch = []
        for token_ids in input_ids:
            token_token = [src_id2word[token_id] for token_id in token_ids]
            token_token_batch.append(token_token)
        return token_token_batch

    def decode_tgt_token_ids(self, pred_ids, tgt_id2word):
        token_token_batch = []

        for token_ids in pred_ids:
            token_token = [tgt_id2word[token_id] for token_id in token_ids]
            token_token_batch.append(token_token)
        return token_token_batch

    def decoding_from_result(self, src_id2word, tgt_id2word,
                             src_seqs, y_pred, dec_output):

        list_input_ids = src_seqs.tolist()
        list_pred_ids = y_pred.max(dim=-1)[1].tolist()

        input_token = self.decode_src_token_ids(list_input_ids, src_id2word)
        pred_token = self.decode_tgt_token_ids(list_pred_ids, tgt_id2word)

        if dec_output is not None:
            real_token = self.decode_tgt_token_ids(dec_output.tolist(), tgt_id2word)

        else :
            # 핑퐁의 띄어쓰기 교정기 적용
            spacer = ChatSpace()
            pred_str = ''.join([token.split('/')[0] for token in pred_token[0][:-1]])
            real_token = spacer.space(pred_str)

        return input_token, pred_token, real_token