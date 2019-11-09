import argparse
import torch
import random
import numpy as np
import sys
import codecs
import os
import learn.tools as tools
import copy
import torch.optim as optim
from collections import defaultdict
from constants import *
from torch.utils.data import DataLoader, Dataset
import time
import itertools

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def read_one_file(file_path):
    # debug
    # print(file_path)
    instances = []
    # with codecs.open(file_path, 'r', 'ISO-8859-2') as fp:
    with open(file_path, 'r', encoding='ISO-8859-2') as fp:
        for line in fp:
            line = line.strip()
            # print(line)
            if line != '':

                columns = line.split("\t")
                text = columns[1]
                label = columns[0]
                tokens = [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]
                instance = {'label':label, 'text': tokens}
                instances.append(instance)

    return instances

def load_data(args):
    # domain apparel, train instances, test instances
    all_data = []

    domain = None
    train_instances = []
    test_instances = []
    file_list = sorted(os.listdir(args.data_path))
    for file_name in file_list:
        current_domain = file_name[:file_name.find(".")]
        if current_domain != domain:
            if len(train_instances) != 0:
                data['train'] = train_instances
            if len(test_instances) != 0:
                data['test'] = test_instances

            domain = current_domain
            data = {'domain':domain}
            all_data.append(data)


        current_type = file_name[file_name.rfind(".")+1:]
        if current_type == 'train':
            train_instances = read_one_file(os.path.join(args.data_path, file_name))
        elif current_type == 'test':
            test_instances = read_one_file(os.path.join(args.data_path, file_name))

    if len(train_instances) != 0:
        data['train'] = train_instances
    if len(test_instances) != 0:
        data['test'] = test_instances

    return all_data

def load_vocab_dict(args, all_data):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()

    for data in all_data:
        train_instances = data['train']
        for instance in train_instances:
            tokens = instance['text']
            for token in tokens:
                vocab.add(token)
        test_instances = data['test']
        for instance in test_instances:
            tokens = instance['text']
            for token in tokens:
                vocab.add(token)

    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}

    w2ind = {w: i for i, w in ind2w.items()}

    return ind2w, w2ind

def load_full_codes(args, all_data):

    ind2c = {0:"0", 1:"1"}

    c2ind = {c: i for i, c in ind2c.items()}

    return ind2c, c2ind


def load_lookups(args, all_data):

    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(args, all_data)

    #get code and description lookups

    ind2c, c2ind = load_full_codes(args, all_data)

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind}

    return dicts

def prepare_instance(dicts, args, data_points):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    for data_point in data_points:

        text = data_point['text']
        label = data_point['label']

        labels_idx = np.zeros(num_labels)
        code = int(c2ind[label])
        labels_idx[code] = 1

        tokens = []
        tokens_id = []
        for token in text:
            tokens.append(token)
            token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
            tokens_id.append(token_id)

        if len(tokens) > MAX_LENGTH:
            tokens = tokens[:MAX_LENGTH]
            tokens_id = tokens_id[:MAX_LENGTH]


        dict_instance = {'label': labels_idx, 'tokens': tokens, "tokens_id": tokens_id}

        instances.append(dict_instance)

    return instances

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def pad_sequence(x, max_len, type=np.int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x

def my_collate2(x):

    words = [x_['tokens_id'] for x_ in x]

    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)

    labels = [x_['label'] for x_ in x]

    positions = [list(range(1, len+1)) for len in seq_len]
    positions = pad_sequence(positions, max_seq_len)

    # text_inputs = [x_['tokens'] for x_ in x]
    text_inputs = [x_['tokens']+ ['<pad>']* (max_seq_len - len(x_['tokens'])) for x_ in x]

    return inputs_id, labels, positions, text_inputs


def init(args):
    all_data = load_data(args)

    dicts = load_lookups(args, all_data)

    if args.mode == 'stl':
        m = tools.pick_model(args, dicts)
        print(m)

        models = []
        optimizers = []
        for data in all_data:
            model = copy.deepcopy(m)
            models.append(model)

            optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
            optimizers.append(optimizer)

            if args.tune_wordemb == False:
                model.freeze_net()

        feature_extractor = None
    elif args.mode == 'fs-mtl':
        model = tools.pick_model(args, dicts)
        print(model)

        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        if args.tune_wordemb == False:
            model.freeze_net()

        feature_extractor = None
        models = [model]
        optimizers = [optimizer]
    elif args.mode == 'sp-mtl':
        feature_extractor, models = tools.pick_model1(args, dicts, all_data)

        optimizer = optim.Adam(itertools.chain(
            *map(list, [feature_extractor.parameters()] + [m.parameters() for m in models])),
                               weight_decay=args.weight_decay, lr=args.lr)

        optimizers = [optimizer]
    else:
        raise RuntimeError("wrong mode")

    return args, models, optimizers, None, dicts, all_data, feature_extractor

def train(args, model, optimizer, gpu, data_loader):

    losses = 0
    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):

        inputs_id, labels, positions, text_inputs = next(data_iter)

        inputs_id, labels, positions, text_inputs = torch.LongTensor(inputs_id), torch.FloatTensor(labels), \
                                                   torch.LongTensor(positions), text_inputs

        if gpu >= 0:
            inputs_id, labels, positions, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), positions.cuda(gpu), text_inputs

        output, loss = model(inputs_id, labels, positions, text_inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses

def test(args, model, gpu, data_loader, feature_extractor=None):

    model.eval()
    if args.mode == 'sp-mtl':
        feature_extractor.eval()
    correct_ct = 0
    total_ct = 0

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():

            inputs_id, labels, positions, text_inputs = next(data_iter)

            inputs_id, labels, positions, text_inputs = torch.LongTensor(inputs_id), torch.FloatTensor(labels), \
                                                        torch.LongTensor(positions), text_inputs

            if gpu >= 0:
                inputs_id, labels, positions, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), positions.cuda(
                    gpu), text_inputs

            if args.mode == 'fs-mtl' or args.mode == 'stl':
                output, _ = model(inputs_id, labels, positions, text_inputs)
            elif args.mode == 'sp-mtl':
                hidden = feature_extractor(inputs_id, labels, positions, text_inputs)
                output, _ = model(hidden, labels, positions, text_inputs)

            # output = torch.sigmoid(output)
            # output = output.data.cpu().numpy()
            # target_data = labels.data.cpu().numpy()
            # output = np.round(output)

            _, pred = torch.max(output, 1)
            _, gold = torch.max(labels, 1)
            correct = (pred == gold).sum().item()

            correct_ct += correct
            total_ct += gold.size(0)

    accuracy = correct_ct*1.0/total_ct
    return accuracy

def train_epochs(args, model, optimizer, params, dicts, data):
    print("######## begin train {}".format(data['domain']))

    train_instances = prepare_instance(dicts, args, data['train'])
    if args.use_devel:
        dev_instance = train_instances[-200:]
        print("dev_instance {} in domain {}".format(len(dev_instance), data['domain']))
        train_instances = train_instances[:-200]
    print("train_instances {} in domain {}".format(len(train_instances), data['domain']))
    test_instances = prepare_instance(dicts, args, data['test'])
    print("test_instances {} in domain {}".format(len(test_instances), data['domain']))

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=my_collate2)
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate2)

    best_test = -10
    bad_counter = 0

    for epoch in range(args.n_epochs):

        losses = train(args, model, optimizer, args.gpu, train_loader)
        accuracy = test(args, model, args.gpu, test_loader)
        print('epoch %d finished, total loss %.4f, test accuracy %.4f' % (epoch, losses, accuracy))

        if accuracy > best_test:
            print("Exceed previous best performance: %.4f" % (best_test))
            best_test = accuracy
            bad_counter = 0

        else:
            bad_counter += 1

        sys.stdout.flush()

        if bad_counter >= args.patience:
            print('Early Stop!')
            break

    print("######## end train {}".format(data['domain']))
    return best_test

def endless_get_next_batch(loaders, iters, domain):
    try:
        inputs_id, labels, positions, text_inputs = next(iters[domain])
    except StopIteration:
        iters[domain] = iter(loaders[domain])
        inputs_id, labels, positions, text_inputs = next(iters[domain])

    return inputs_id, labels, positions, text_inputs

def train_mtl(args, models, optimizers, dicts, all_data, feature_extractor):
    train_loaders, train_iters, test_loaders = {}, {}, {}

    for data in all_data:

        train_instances = prepare_instance(dicts, args, data['train'])
        if args.use_devel:
            dev_instance = train_instances[-200:]
            print("dev_instance {} in domain {}".format(len(dev_instance), data['domain']))
            train_instances = train_instances[:-200]
        print("train_instances {} in domain {}".format(len(train_instances), data['domain']))
        test_instances = prepare_instance(dicts, args, data['test'])
        print("test_instances {} in domain {}".format(len(test_instances), data['domain']))

        train_loaders[data['domain']] = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn = my_collate2)
        train_iters[data['domain']] = iter(train_loaders[data['domain']])
        test_loaders[data['domain']] = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate2)


    best_test = -10
    bad_counter = 0

    for epoch in range(args.n_epochs):

        if args.mode == 'fs-mtl':
            model = models[0]
            model.train()
            optimizer = optimizers[0]
        elif args.mode == 'sp-mtl':
            feature_extractor.train()
            for model in models:
                model.train()
            optimizer = optimizers[0]

        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[all_data[0]['domain']])
        losses = 0

        for i in range(num_iter):

            for domain_idx, data in enumerate(all_data):
                inputs_id, labels, positions, text_inputs = endless_get_next_batch(train_loaders, train_iters, data['domain'])
                inputs_id, labels, positions, text_inputs = torch.LongTensor(inputs_id), torch.FloatTensor(labels), torch.LongTensor(positions), text_inputs
                if args.gpu >= 0:
                    inputs_id, labels, positions, text_inputs = inputs_id.cuda(args.gpu), labels.cuda(args.gpu), positions.cuda(args.gpu), text_inputs

                if args.mode == 'fs-mtl':
                    output, loss = model(inputs_id, labels, positions, text_inputs)
                elif args.mode == 'sp-mtl':
                    hidden = feature_extractor(inputs_id, labels, positions, text_inputs)
                    output, loss = models[domain_idx](hidden, labels, positions, text_inputs)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

        average = 0
        score_of_each_domain = {}
        for domain_idx, data in enumerate(all_data):
            if args.mode == 'fs-mtl':
                accuracy = test(args, model, args.gpu, test_loaders[data['domain']], feature_extractor)
            elif args.mode == 'sp-mtl':
                accuracy = test(args, models[domain_idx], args.gpu, test_loaders[data['domain']], feature_extractor)
            average += accuracy
            score_of_each_domain[data['domain']] = "%.4f" % accuracy
        average = average*1.0/len(all_data)

        print('epoch %d finished, total loss %.4f, average test accuracy %.4f' % (epoch, losses, average))

        if average > best_test:
            print("Exceed previous best performance: %.4f" % (best_test))
            print("score_of_each_domain {}".format(score_of_each_domain))
            best_test = average
            bad_counter = 0

        else:
            bad_counter += 1

        sys.stdout.flush()

        if bad_counter >= args.patience:
            print('Early Stop!')
            break

    return best_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("--vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("--Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "logreg", "saved",
                                                    "conv_attn_ldep", 'bert_conv_attn', 'bert_conv', 'resnet_attn',
                                                    'conv_attn_lco','bert_pooling', 'transformer1', 'transformer2',
                                                    'transformer3', 'transformer4', 'bert_seq_cls',
                                                    'CNN', 'TFIDF', 'MultiCNN', 'ResCNN', 'MultiResCNN'], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")

    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)", dest='cell_type',
                        default='gru')
    parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
                        help="optional flag for rnn to use a bidirectional model")
    parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=4,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--pool", choices=['max', 'avg'], required=False, dest="pool", help="which type of pooling to do (logreg model only)")
    parser.add_argument("--code-emb", type=str, required=False, dest="code_emb",
                        help="point to code embeddings to use for parameter initialization, if applicable")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--lmbda", type=float, required=False, dest="lmbda", default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3', required=False,
                        help="version of MIMIC in use (default: mimic3)")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", type=int, default=-1, required=False,
                        help="optional flag to use GPU if available")
    parser.add_argument("--public-model", dest="public_model", action="store_const", required=False, const=True,
                        help="optional flag for testing pre-trained models from the public github")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    parser.add_argument("--bert_dir", dest="bert_dir", type=str)
    parser.add_argument("--bert_chunk_len", dest="bert_chunk_len", type=int, default=128)
    parser.add_argument("--conv_layer", dest="conv_layer", type=int, default=1)
    parser.add_argument("--use_res", dest="use_res", action="store_const", const=True, default=False)
    parser.add_argument("--use_ext_emb", dest="use_ext_emb", action="store_const", const=True, default=False)
    parser.add_argument("--use_pos", dest="use_pos", type=str, default=None, help='NN,VBP')
    parser.add_argument('--random_seed', type=int, default=1)

    # sentence encoder initialized from bert, 1-12
    parser.add_argument('--sent_encoder_layer_start', type=int, default=1)
    parser.add_argument('--sent_encoder_layer_end', type=int, default=1)
    parser.add_argument("--max_sent_num", type=int, default=128)

    # document encoder
    parser.add_argument('--doc_encoder_layer', type=int, default=1)

    # CNN
    parser.add_argument("--use_tfidf", dest="use_tfidf", action="store_const", const=True, default=False)
    parser.add_argument("--use_position", dest="use_position", action="store_const", const=True, default=False)
    parser.add_argument("--tune_wordemb", dest="tune_wordemb", action="store_const", const=True, default=False)
    parser.add_argument("--fasttext", type=str, required=False, dest="fasttext", default=None)
    parser.add_argument("--glove", type=str, required=False, dest="glove", default=None)
    parser.add_argument("--weighted_loss", dest="weighted_loss", action="store_const", const=True, default=False)
    parser.add_argument("--mu", dest="mu", type=float, default=0.15)
    parser.add_argument("--elmo", type=str, default=None)

    parser.add_argument("--use_devel", dest="use_devel", action="store_const", const=True, default=False)
    parser.add_argument("--mode", type=str, default='stl', choices=['stl','fs-mtl', 'sp-mtl'])

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command

    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)


    args, models, optimizers, params, dicts, all_data, feature_extractor = init(args)

    if args.mode == 'stl':
        average = 0
        for model, optimizer, data in zip(models, optimizers, all_data):
            accuracy = train_epochs(args, model, optimizer, params, dicts, data)
            average += accuracy
        average = average / len(all_data)
    elif args.mode == 'fs-mtl' or args.mode == 'sp-mtl':
        average = train_mtl(args, models, optimizers, dicts, all_data, feature_extractor)


    print("average score %.4f" % (average))

