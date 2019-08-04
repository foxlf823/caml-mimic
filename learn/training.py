"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import csv
import argparse
import os 
import numpy as np
import operator
import random
import sys
import time
from tqdm import tqdm
from collections import defaultdict

from constants import *
import datasets
import evaluation
import interpret
import persistence
import learn.models as models
import learn.tools as tools
from pytorch_pretrained_bert import BertTokenizer, BertAdam

from torch.utils.data import DataLoader, Dataset

def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    #load vocab and other lookups
    desc_embed = args.lmbda > 0
    print("loading lookups...")
    dicts = datasets.load_lookups(args, desc_embed=desc_embed)

    model = tools.pick_model(args, dicts)
    # print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

from pytorch_pretrained_bert import BertTokenizer

def prepare_instance(dicts, filename, args):
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
    instances = []
    num_labels = len(dicts['ind2c'])

    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=True)

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)

        for row in r:

            text = row[2]
            hadm_id = int(row[1])

            cur_code_set = set()
            labels_idx = np.zeros(num_labels)
            labelled = False

            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    cur_code_set.add(code)
                    labelled = True
            if not labelled:
                continue

            tokens = text.split()
            sentences = []
            sentences_id = []
            sentence = []
            masks = []
            segments = []
            for token in tokens:
                if token == '[CLS]':
                    sentence.append('[CLS]')
                elif token == '[SEP]':
                    if len(sentence) > args.bert_chunk_len:
                        sentence = sentence[:args.bert_chunk_len]
                    sentence.append('[SEP]')
                    sentences.append(sentence)
                    sentence_id = wp_tokenizer.convert_tokens_to_ids(sentence)
                    sentences_id.append(sentence_id)
                    segment = [0] * len(sentence)
                    segments.append(segment)
                    mask = [1] * len(sentence)
                    masks.append(mask)
                    sentence = []
                else:
                    wps = wp_tokenizer.tokenize(token)
                    sentence.extend(wps)

            dict_instance = {'label':labels_idx, 'hadm_id':hadm_id, 'cur_code_set':cur_code_set, 'sentences':sentences,
                             "sentences_id":sentences_id, "segments":segments, "masks":masks}

            instances.append(dict_instance)

    return instances

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x

def my_collate(x):

    sentences_id = x[0]["sentences_id"]
    segments = x[0]["segments"]
    masks = x[0]['masks']

    labels = [x_['label'] for x_ in x]

    lengths = [len(sentence_id) for sentence_id in sentences_id]
    length = max(lengths)

    inputs_id = pad_sequence(sentences_id, length)
    segments = pad_sequence(segments, length)
    masks = pad_sequence(masks, length)

    return inputs_id, segments, masks, labels


def train_epochs(args, model, optimizer, params, dicts):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None

    # prepare instance before epoch since pos tagging is too slow
    train_instances = prepare_instance(dicts, args.data_path, args)
    print("train_instances {}".format(len(train_instances)))
    dev_instances = prepare_instance(dicts, args.data_path.replace('train','dev'), args)
    print("dev_instances {}".format(len(dev_instances)))
    test_instances = prepare_instance(dicts, args.data_path.replace('train','test'), args)
    print("test_instances {}".format(len(test_instances)))
    # train_instances = dev_instances
    # test_instances = dev_instances

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=my_collate)
    dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=my_collate)
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate)

    if not args.test_model and args.model.find("bert") != -1:
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # batch size is 1, one doc one step
        num_train_optimization_steps = int(len(train_instances)) * args.n_epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=0.1,
                             t_total=num_train_optimization_steps)


        print("Warmup steps = %d, Num steps = %d" %
              (int(num_train_optimization_steps * 0.1), num_train_optimization_steps))

    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            #os.mkdir(model_dir)
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
        metrics_all = one_epoch(args, model, optimizer, args.Y, epoch, args.n_epochs, args.batch_size, args.data_path,
                                                  args.version, test_only, dicts, model_dir, 
                                                  args.samples, args.gpu, args.quiet, train_instances, dev_instances, test_instances,
                                train_loader, dev_loader, test_loader)
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        #save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion, evaluate)

        sys.stdout.flush()

        if test_only:
            #we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = tools.pick_model(args, dicts)
    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev': 
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False
        
def one_epoch(args, model, optimizer, Y, epoch, n_epochs, batch_size, data_path, version, testing, dicts, model_dir,
              samples, gpu, quiet, train_instances, dev_instances, test_instances, train_loader, dev_loader, test_loader):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        epoch_start = time.time()
        losses, unseen_code_inds = train(args, model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet,
                                         train_instances, train_loader)
        loss = np.mean(losses)
        epoch_finish = time.time()
        print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish-epoch_start, loss))
    else:
        loss = np.nan
        if model.lmbda > 0:
            #still need to get unseen code inds
            print("getting set of codes not in training set")
            c2ind = dicts['c2ind']
            unseen_code_inds = set(dicts['ind2c'].keys())
            num_labels = len(dicts['ind2c'])
            with open(data_path, 'r') as f:
                r = csv.reader(f)
                #header
                next(r)
                for row in r:
                    unseen_code_inds = unseen_code_inds.difference(set([c2ind[c] for c in row[3].split(';') if c != '']))
            print("num codes not in train set: %d" % len(unseen_code_inds))
        else:
            unseen_code_inds = set()

    fold = 'test' if version == 'mimic2' else 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True
        quiet = False

    #test on dev
    metrics = test(args, model, Y, epoch, data_path, fold, gpu, version, unseen_code_inds, dicts, samples, model_dir,
                   testing, dev_instances, dev_loader)
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(args, model, Y, epoch, data_path, "test", gpu, version, unseen_code_inds, dicts, samples,
                          model_dir, True, test_instances, test_loader)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(args, model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet, instances, data_loader):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)

    losses = []


    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        inputs_id, segments, masks, labels = next(data_iter)

        inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), torch.LongTensor(masks), torch.FloatTensor(labels)

        if gpu >= 0:
            inputs_id, segments, masks, labels = inputs_id.cuda(gpu), segments.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

        output, loss = model(inputs_id, segments, masks, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses, None

def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    #wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    #replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0

def test(args, model, Y, epoch, data_path, fold, gpu, version, code_inds, dicts, samples, model_dir, testing,
         instances, data_loader):
    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    #initialize stuff for saving attention samples
    if samples:
        tp_file = open('%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open('%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        window_size = model.conv.weight.data.size()[2]

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    model.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():
            inputs_id, segments, masks, labels = next(data_iter)

            inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(
                segments), torch.LongTensor(masks), torch.FloatTensor(labels)

            if gpu >= 0:
                inputs_id, segments, masks, labels = inputs_id.cuda(gpu), segments.cuda(gpu), masks.cuda(
                    gpu), labels.cuda(gpu)

            output, loss = model(inputs_id, segments, masks, labels)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            #save predictions, target, hadm ids
            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)


    #close files if needed
    if samples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    # preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 50 else [8,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "logreg", "saved",
                                                    "conv_attn_ldep", 'bert_conv_attn', 'bert_conv', 'resnet_attn',
                                                    'conv_attn_lco','bert_pooling'], help="model")
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

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command

    if args.use_pos:
        pos_tags = args.use_pos.split(",")
        args.use_pos = pos_tags

    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)

