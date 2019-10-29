# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import editdistance

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelTextData
from module import GruEncoder, GruDecoder, EncoderRNN, Decoder, LuongDecoder
from module import Seq2Seq
from module.embedding import make_fasttext_embedding_vocab_weight
from module.preprocess import MecabTokenizer
from module.preprocess import NltkTokenizer
from util import AttributeDict
from util import get_device
from module.evaluate_transformer import token_decoder

train_config = AttributeDict({
    "n_epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "src_tokenizer": NltkTokenizer,
    "tgt_tokenizer": NltkTokenizer,
    "src_vocab_filename": "src_vocab.txt",
    "src_word_embedding_filename": "src_word_embedding.npy",
    "tgt_vocab_filename": "tgt_vocab.txt",
    "tgt_word_embedding_filename": "tgt_word_embedding.npy",
    "train_src_corpus_filename": "korean-english-park.train.ko",
    "train_tgt_corpus_filename": "korean-english-park.train.en",
    "dev_src_corpus_filename": "korean-english-park.dev.ko",
    "dev_tgt_corpus_filename": "korean-english-park.dev.en",
})

encoder_params = AttributeDict({
    "embedding_dim": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout_prob": 0.3,
    "bidirectional": True,
    "max_seq_len": 100,
})

decoder_params = AttributeDict({
    "embedding_dim": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout_prob": 0.3,
    "max_seq_len": 100,
})


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


config = {
  "decoder": "Loung",
  "encoder": "RNN",
  "n_channels": 4,
  "encoder_hidden": 256,
  "encoder_layers": 2,
  "encoder_dropout": 0.1,
  "bidirectional_encoder": True,
  "decoder_hidden": 256,
  "decoder_layers": 2,
  "decoder_dropout": 0.1,
  #"vocab_size":dataset.VOCAB_SIZE+3 , # TopyDataset 의 vocab 사이즈는 Encoder, Decoder 구분없이 같음
  "batch_size": 32,
  "embedding_dim": 64,
  #"attention_score": "concat",
  "attention_score": "general",
  "learning_rate": 0.001,
  "gpu": True,
  "loss": "cross_entropy"
}


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


def train_model(model: nn.Module,
                optimizer,
                loss_func,
                data_loader: DataLoader,
                device,
                train_config: AttributeDict,
                epoch: int):
    # Set train flag
    model.train()
    n_epochs = train_config.n_epochs
    losses = []
    data_length = len(data_loader)

    for _, batch in enumerate(tqdm(data_loader, total=data_length, desc=f'Epoch {epoch:3}')):
        src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
        src_seqs = src_seqs.to(device)
        src_lengths = src_lengths.to(device)
        tgt_seqs = tgt_seqs.to(device)
        tgt_lengths = tgt_lengths.to(device)
        logits, labels, predictions = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)

        loss = loss_func(logits, labels)
        losses.append(loss.item())

        # initialize buffer
        optimizer.zero_grad()

        # calculate gradient
        loss.backward()

        # update model parameter
        optimizer.step()

    print(f'Epochs [{epoch}/{n_epochs}] avg losses: {np.mean(losses):05.3f}')
    return


def train(model, optimizer, train_loader, epoch, n_epochs, src_id2word, tgt_id2word):
    losses = []
    cers = []

    model.train()  # train mode
    count = 0
    n_iter = 0
    for batch in train_loader:

        loss, logits, labels, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        optimizer.step()

        n_iter += 1  # count number of iteration

        #if n_iter % 10 == 0:  # print loss only if it's training stage
        #    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch

        #    loss, logits, labels, alignments, inference = model.loss(batch)
        #    y_pred_copy = logits.detach()
        #    y_pred_copy = y_pred_copy.reshape(src_seqs.size(0), 100, len(tgt_id2word))

            #y_pred = logits
            #y_pred_copy = y_pred.detach()
        #    dec_output_copy = tgt_seqs.detach()
        #    get_sentence = token_decoder()
        #    input_token, pred_token, real_token = get_sentence.decoding_from_result(src_id2word, tgt_id2word,
        #                                                                        src_seqs, y_pred_copy,
        #                                                                        dec_output_copy)

        #    print("input: ", input_token[0])
        #    print("pred: ", pred_token[0])
        #    print("real: ", real_token[0])

        if n_iter % 10 == 0:  # print loss only if it's training stage
            print('\n [{}] current_iter_loss= {:05.3f}'.format(n_iter, loss))

    print('\n [{}/{}] avg_loss= {:05.3f}'.format(epoch, n_epochs, np.mean(losses)))

    return model, optimizer


def evaluate(model, eval_loader, src_id2word, tgt_id2word):
    losses = []
    accs = []
    edits = []

    model.eval()

    with torch.no_grad():
        n_iter = 0
        for batch in eval_loader:
            # t.set_description(" Evaluating... (train={})".format(model.training))
            src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
            loss, logits, labels, alignments, _ = model.loss(batch)
            preds = logits.detach().cpu().numpy()

            #print("Predi", preds[0])
            #print("Label", labels[0])
            id2word: dict = {}
            #def vec2sent(vec: torch.Tensor):
            #    vec.argmax(dim=-1)
            #    sent = []
            #    for id in vec:
            #        sent.append(id2word[id])
            #    return sent

            n_iter += 1  # count number of iteration
            if n_iter % 1000 == 0:  # print loss only if it's training stage

                #loss, logits, labels, alignments, inference = model.loss(batch)
                y_pred_copy = logits.detach()
                y_pred_copy = y_pred_copy.reshape(src_seqs.size(0), 100, len(tgt_id2word))

                dec_output_copy = tgt_seqs.detach()
                get_sentence = token_decoder()
                input_token, pred_token, real_token = get_sentence.decoding_from_result(src_id2word, tgt_id2word,
                                                                                    src_seqs, y_pred_copy,
                                                                                    dec_output_copy)

                print("input: ", input_token[0])
                print("pred: ", pred_token[0])
                print("real: ", real_token[0])

            acc = 100 * np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            edit = editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)

            losses.append(loss.item())

            accs.append(acc)
            edits.append(edit)

        align = alignments.detach().cpu().numpy()[:, :, 0]

    print("  End of evaluation : loss {:05.3f} , acc {:03.1f} , edits {:03.3f}".format(np.mean(losses), np.mean(accs),
                                                                                       np.mean(edits)))

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
    dev_src_corpus_file_path = os.path.join(dataset_dir, train_config.train_src_corpus_filename)
    dev_tgt_corpus_file_path = os.path.join(dataset_dir, train_config.train_tgt_corpus_filename)

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

    config["src_vocab_size"] = len(src_word2id)
    config["tgt_vocab_size"] = len(tgt_word2id)

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

    dev_dataset = ParallelTextData(src_tokenizer,
                               tgt_tokenizer,
                               dev_src_corpus_file_path,
                               dev_tgt_corpus_file_path,
                               encoder_params.max_seq_len,
                               decoder_params.max_seq_len,
                               src_word2id,
                               tgt_word2id)
    dev_data_loader = DataLoader(dev_dataset,
                             batch_size=train_config.batch_size,
                             shuffle=True,
                             collate_fn=dataset.collate_func)

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
    batch_size = 32
    #epochs = 5
    #model = Seq2Seq(config, len(src_word2id), len(tgt_word2id))
    model = Seq2Seq(config)
    if config["gpu"]:
        model = model.cuda()

    print("BB", config.get("attention_score", "dot"))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    for epoch in range(train_config["n_epochs"]):
        model, optimizer = train(model, optimizer, data_loader, epoch, train_config["n_epochs"], src_id2word, tgt_id2word)
        evaluate(model, dev_data_loader, src_id2word, tgt_id2word)
if __name__ == '__main__':
    main()
