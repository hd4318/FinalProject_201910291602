# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import torch
import nltk
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset import ParallelTextData
from module import GruEncoder, GruDecoder#, EncoderRNN, Decoder, LuongDecoder
from module.decodermodel import EncoderRNN, Decoder, LuongDecoder
from module import Seq2Seq
from module.embedding import make_fasttext_embedding_vocab_weight
from module.tokenizer import MecabTokenizer, NltkTokenizer

from model.optim import GradualWarmupScheduler
from evaluate_transformer import token_decoder, evaluate
from util.tokens import PAD_TOKEN_ID
from params import decoder_params, encoder_params, eval_params, train_config, config
from util.util import AttributeDict
from util.util import get_checkpoint_dir_path
from util.util import get_device
from util.util import train_step
from util.util import Config, CheckpointManager, SummaryManager

def check_config(config: AttributeDict):
    assert isinstance(config.get('learning_rate'), float), \
        'learning_rate should be float value.'
    assert config.get('src_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'src_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('tgt_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'tgt_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('src_vocab_filename', None) is not None, \
        'src_vocab_filename must not be None'
    assert config.get('tgt_vocab_filename', None) is not None, \
        'tgt_vocab_filename must not be None'
    assert config.get('src_word_embedding_filename', None) is not None, \
        'src_word_embedding_filename must not be None'
    assert config.get('tgt_word_embedding_filename', None) is not None, \
        'tgt_word_embedding_filename must not be None'
    assert config.get('train_src_corpus_filename', None) is not None, \
        'train_src_corpus_filename must not be None'
    assert config.get('train_tgt_corpus_filename', None) is not None, \
        'train_tgt_corpus_filename must not be None'


def ensure_vocab_embedding(
        tokenizer,
        vocab_file_path: str,
        word_embedding_file_path: str,
        corpus_file_path: str,
        embedding_dimen: int,
        tag: str,
):
    """
    :return: (word2id, id2word)
    """
    if not os.path.exists(vocab_file_path) or not os.path.exists(word_embedding_file_path):
        # Make source embedding
        print(f'{tag} embedding information is not exists.')

        embedding = make_fasttext_embedding_vocab_weight(
            tokenizer,
            corpus_file_path=corpus_file_path,
            vocab_path=vocab_file_path,
            weight_path=word_embedding_file_path,
            embedding_dim=embedding_dimen,
        )
        print(f'{tag} vocab size: {embedding.vocab_size}')

    with open(vocab_file_path, mode='r', encoding='utf-8') as f:
        tokens = f.readlines()
    word2id = {}
    id2word = {}
    for index, token in enumerate(tokens):
        token = token.strip()
        if len(token) == 0:
            continue
        word2id[token] = index
        id2word[index] = token

    embedding_matrix = np.load(word_embedding_file_path)

    return word2id, id2word, embedding_matrix


# def train_model(model: nn.Module,
#                 optimizer,
#                 loss_func,
#                 data_loader: DataLoader,
#                 device,
#                 train_config: AttributeDict,
#                 epoch: int):
#     # Set train flag
#     model.train()
#     n_epochs = train_config.n_epochs
#     losses = []
#     data_length = len(data_loader)
#
#     for _, batch in enumerate(tqdm(data_loader, total=data_length, desc=f'Epoch {epoch:3}')):
#         src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
#         src_seqs = src_seqs.to(device)
#         src_lengths = src_lengths.to(device)
#         tgt_seqs = tgt_seqs.to(device)
#         tgt_lengths = tgt_lengths.to(device)
#         logits, labels, predictions = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)
#
#         loss = loss_func(logits, labels)
#         losses.append(loss.item())
#
#         # initialize buffer
#         optimizer.zero_grad()
#
#         # calculate gradient
#         loss.backward()
#
#         # update model parameter
#         optimizer.step()
#
#     print(f'Epochs [{epoch}/{n_epochs}] avg losses: {np.mean(losses):05.3f}')
#     return


def train(model, optimizer, train_loader, n_epochs, device, val_data_loader, src_id2word, tgt_id2word, loss_func):
    losses = []
    cers = []

    # save
    try:
        if not os.path.exists('./model'):
            os.makedirs('./model')
    except OSError:
        print('Error: Creating model directory')

    model_dir = Path('./model')
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10
    best_tr_loss = 1e+10


    # load
    if (model_dir / 'best.tar').exists():
        print("pretrained model exists")
        checkpoint = checkpoint_manager.load_checkpoint('best.tar')
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(n_epochs):
        tr_loss = 0
        model.train()
        n_iter = 0
        for step, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch:3}')):
            optimizer.zero_grad()
            src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch

            loss, logits, labels, alignments, inference = model.loss(batch)
            y_pred_copy = logits.detach()
            y_pred_copy = y_pred_copy.reshape(src_seqs.size(0), 100, len(tgt_id2word))
            # y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output_copy = tgt_seqs.detach()
            dec_output = tgt_seqs.view(-1).long()

            # padding 제외한 value index 추출
            real_value_index = [dec_output != 0]

            # print(preds[0])
            losses.append(loss.item())

            # Reset gradients
            optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            optimizer.step()
            n_iter += 1  # count number of iteration
            if n_iter % 10 == 0:  # print loss only if it's training stage
                print('\n [{}] current_iter_loss= {:05.3f}'.format(n_iter, loss))

            tr_loss += loss.item()
            tr_loss_avg = tr_loss / (step + 1)
            tr_summary = {'loss': tr_loss_avg}
            total_step = epoch * len(train_loader) + step

            # Eval
            if total_step % 100 == 0 and total_step != 0:
                print("train: ")

            get_sentence = token_decoder()
            input_token, pred_token, real_token = get_sentence.decoding_from_result(src_id2word, tgt_id2word,
                                                                                    src_seqs, y_pred_copy,
                                                                                    dec_output_copy)

            # print("input: ", input_token)
            # print("pred: ", pred_token)
            # print("real: ", real_token)

            bleu_scores = []
            for j in range(len(real_token)):
                bleu_score = nltk.translate.bleu_score.corpus_bleu([real_token[j]], [pred_token[j]], weights=[1])
                bleu_scores.append(bleu_score)

            bleu_avg = np.array(bleu_scores).mean()
            # print('train bleu: ', bleu_avg)

        model.eval()
        # print("eval: ")
        val_summary = evaluate(model, val_data_loader, {'loss': loss_func}, device)
        val_loss = val_summary['loss']

        tqdm.write('epoch : {}, step : {}, '
                   'tr_loss: {:.3f}, val_loss: {:.3f}'.format(epoch + 1, total_step,
                                                              tr_summary['loss'], val_summary['loss']))
        is_best = val_loss < best_val_loss  # loss 기준

        if is_best:
            print("[Best model Save] train_loss: {}, val_loss: {}".format(tr_summary['loss'], val_loss))
            # CPU에서도 동작 가능하도록 자료형 바꾼 뒤 저장
            state = {'epoch': epoch + 1,
                     'model_state_dict': model.to(torch.device('cpu')).state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
            summary = {'train': tr_summary}

            summary_manager.update(summary)
            summary_manager.save('summary.json')
            checkpoint_manager.save_checkpoint(state, 'best.tar')

        model.to(device)
        model.train()
    ################################################################
    # model.train()  # train mode
    # count = 0
    # n_iter = 0
    # for batch in train_loader:
    #     loss, logits, labels, alignments, inference = model.loss(batch)
    #     preds = logits.detach().cpu().numpy()
    #
    #     # print(preds[0])
    #     losses.append(loss.item())
    #     # Reset gradients
    #     optimizer.zero_grad()
    #     # Compute gradients
    #     loss.backward()
    #     optimizer.step()
    #     n_iter += 1  # count number of iteration
    #     if n_iter % 10 == 0:  # print loss only if it's training stage
    #         print('\n [{}] current_iter_loss= {:05.3f}'.format(n_iter, loss))
    #
    # print('\n [{}/{}] avg_loss= {:05.3f}'.format(epoch, n_epochs, np.mean(losses)))
    ################################################################
    torch.save(model, './model/trained_model')
    return model, optimizer

def main():
    print("***** Train start *****")
    tokenizer = MecabTokenizer()

    check_config(train_config)

    device = get_device()
    print(f'  Available device is {device}')

    src_tokenizer = train_config.src_tokenizer()
    tgt_tokenizer = train_config.tgt_tokenizer()

    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')

    src_vocab_file_path = os.path.join(dataset_dir, train_config.src_vocab_filename)
    tgt_vocab_file_path = os.path.join(dataset_dir, train_config.tgt_vocab_filename)
    src_word_embedding_file_path = os.path.join(dataset_dir,
                                                train_config.src_word_embedding_filename)
    tgt_word_embedding_file_path = os.path.join(dataset_dir,
                                                train_config.tgt_word_embedding_filename)
    train_src_corpus_file_path = os.path.join(dataset_dir, train_config.train_src_corpus_filename)
    train_tgt_corpus_file_path = os.path.join(dataset_dir, train_config.train_tgt_corpus_filename)

    src_word2id, src_id2word, src_embed_matrix = ensure_vocab_embedding(
        src_tokenizer,
        src_vocab_file_path,
        src_word_embedding_file_path,
        train_src_corpus_file_path,
        encoder_params.embedding_dim,
        "Source")

    tgt_word2id, tgt_id2word, tgt_embed_matrix = ensure_vocab_embedding(
        tgt_tokenizer,
        tgt_vocab_file_path,
        tgt_word_embedding_file_path,
        train_tgt_corpus_file_path,
        decoder_params.embedding_dim,
        "Target")

    dataset = ParallelTextData(src_tokenizer,
                               tgt_tokenizer,
                               train_src_corpus_file_path,
                               train_tgt_corpus_file_path,
                               encoder_params.max_seq_len,
                               decoder_params.max_seq_len,
                               src_word2id,
                               tgt_word2id)
    data_loader = DataLoader(dataset,
                             batch_size=train_config.batch_size,
                             shuffle=True,
                             collate_fn=dataset.collate_func)

    # validation data set
    val_src_corpus_file_path = os.path.join(dataset_dir, train_config.val_src_corpus_filename)
    val_tgt_corpus_file_path = os.path.join(dataset_dir, train_config.val_tgt_corpus_filename)

    val_src_word2id, val_src_id2word, val_src_embed_matrix = ensure_vocab_embedding(
            src_tokenizer,
            src_vocab_file_path,
            src_word_embedding_file_path,
            val_src_corpus_file_path,
            encoder_params.embedding_dim,
            "Val Source")

    val_tgt_word2id, val_tgt_id2word, val_tgt_embed_matrix = ensure_vocab_embedding(
            tgt_tokenizer,
            tgt_vocab_file_path,
            tgt_word_embedding_file_path,
            val_tgt_corpus_file_path,
            decoder_params.embedding_dim,
            "Val Target")
    print('Val target len', len(tgt_word2id))

    val_dataset = ParallelTextData(src_tokenizer,
                                   tgt_tokenizer,
                                   val_src_corpus_file_path,
                                   val_tgt_corpus_file_path,
                                   encoder_params.max_seq_len,
                                   decoder_params.max_seq_len,
                                   val_src_word2id,
                                   val_tgt_word2id)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=train_config.batch_size,
                                 shuffle=True,
                                 collate_fn=dataset.collate_func
                              )
    #encoder = GruEncoder(vocab_size=len(src_word2id),
    #                     embedding_dim=encoder_params.embedding_dim,
    #                     hidden_size=encoder_params.hidden_size,
    #                     bidirectional=encoder_params.bidirectional,
    #                     num_layers=encoder_params.num_layers,
    #                     dropout_prob=encoder_params.dropout_prob,
    #                     device=device)
    #encoder = EncoderRNN(config, vocab_size=len(src_word2id))
    # Freeze word embedding weight
    #encoder.init_embedding_weight(src_embed_matrix)

    #decoder = GruDecoder(vocab_size=len(tgt_word2id),
    #                     embedding_dim=decoder_params.embedding_dim,
    #                     hidden_size=decoder_params.hidden_size,
    #                     num_layers=decoder_params.num_layers,
    #                     dropout_prob=decoder_params.dropout_prob,
    #                     device=device)
    #decoder = LuongDecoder(config, vocab_size=len(tgt_word2id))
    # Freeze word embedding weight
    #decoder.init_embedding_weight(tgt_embed_matrix)

    # model = Seq2Seq(config, len(src_word2id), len(tgt_word2id))

    # loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    #
    # for epoch in range(train_config.n_epochs):
    #     train_model(model, optimizer, loss_func, data_loader, device, train_config,
    #                 epoch + 1)

    config["src_vocab_size"] = len(src_word2id)
    config["tgt_vocab_size"] = len(tgt_word2id)
    epochs = 5
    model = Seq2Seq(config)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)


    model, optimizer = train(model, optimizer, data_loader, epochs, device, val_data_loader, src_id2word, tgt_id2word, loss_func)
    #     evaluate(model, eval_loader)

if __name__ == '__main__':
    main()
