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
from pytorch_pretrained_bert import BertTokenizer

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
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts

import nltk
def prepare_instance(dicts, filename):
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
    instances = []
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)

        for row in r:

            text = row[2]

            original_text = text.split()
            text = [int(w2ind[w]) if w in w2ind else len(w2ind) + 1 for w in original_text]
            if args.use_pos:
                pos_tags = nltk.pos_tag(original_text)
                # salience
                #pos_tags = [1 if pos_tag in args.use_pos else 0 for _, pos_tag in pos_tags]
                pos_tags = [1 if pos_tag in args.use_pos else 2 for _, pos_tag in pos_tags]
            # truncate long documents
            if len(text) > MAX_LENGTH:
                text = text[:MAX_LENGTH]
                if args.use_pos:
                    pos_tags = pos_tags[:MAX_LENGTH]

            if args.use_pos:
                instances.append({'row':row, 'word':text, 'pos':pos_tags})
            else:
                instances.append({'row': row, 'word': text})

    return instances



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
    train_instances = prepare_instance(dicts, args.data_path)
    print("train_instances {}".format(len(train_instances)))
    dev_instances = prepare_instance(dicts, args.data_path.replace('train','dev'))
    print("dev_instances {}".format(len(dev_instances)))
    test_instances = prepare_instance(dicts, args.data_path.replace('train','test'))
    print("test_instances {}".format(len(test_instances)))


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
                                                  args.samples, args.gpu, args.quiet, train_instances, dev_instances, test_instances)
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
              samples, gpu, quiet, train_instances, dev_instances, test_instances):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        epoch_start = time.time()
        losses, unseen_code_inds = train(args, model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet, train_instances)
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
                   testing, dev_instances)
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(args, model, Y, epoch, data_path, "test", gpu, version, unseen_code_inds, dicts, samples,
                          model_dir, True, test_instances)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(args, model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet, instances):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])

    losses = []
    #how often to print some info to stdout
    print_every = 25

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    unseen_code_inds = set(ind2c.keys())
    desc_embed = model.lmbda > 0

    model.train()
    if args.model.find("bert") != -1:
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    else:
        tokenizer = None
    gen = datasets.data_generator(instances, args, tokenizer, data_path, dicts, batch_size, num_labels, version=version, desc_embed=desc_embed)
    #for batch_idx, tup in tqdm(enumerate(gen)):
    for batch_idx, tup in enumerate(gen):
        data, target, _, code_set, descs = tup
        if args.model.find("bert") != -1:
            word, mask, segment, batch_size, chunk_num = data
            data, target = (torch.LongTensor(word), torch.LongTensor(mask), torch.LongTensor(segment), batch_size, chunk_num), torch.FloatTensor(target)
        else:
            if args.use_pos:
                word, pos = data
                data, target = (torch.LongTensor(word), torch.LongTensor(pos)), torch.FloatTensor(target)
            else:
                data, target = torch.LongTensor(data), torch.FloatTensor(target)
        unseen_code_inds = unseen_code_inds.difference(code_set)
        if gpu >= 0:
            if args.model.find("bert") != -1:
                word, mask, segment, batch_size, chunk_num = data
                data = (word.cuda(gpu), mask.cuda(gpu), segment.cuda(gpu), batch_size, chunk_num)
                target = target.cuda(gpu)
            else:
                if args.use_pos:
                    word, pos = data
                    data = (word.cuda(gpu), pos.cuda(gpu))
                    target = target.cuda(gpu)
                else:
                    data = data.cuda(gpu)
                    target = target.cuda(gpu)
        optimizer.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        output, loss, _ = model(data, target, desc_data=desc_data)

        loss.backward()
        optimizer.step()

        #losses.append(loss.data[0])
        losses.append(loss.item())

        if not quiet and batch_idx % print_every == 0:
            #print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses, unseen_code_inds

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

def test(args, model, Y, epoch, data_path, fold, gpu, version, code_inds, dicts, samples, model_dir, testing, instances):
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

    desc_embed = model.lmbda > 0
    if desc_embed and len(code_inds) > 0:
        unseen_code_vecs(model, code_inds, dicts, gpu)

    model.eval()
    if args.model.find("bert") != -1:
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    else:
        tokenizer = None
    gen = datasets.data_generator(instances, args, tokenizer, filename, dicts, 1, num_labels, version=version, desc_embed=desc_embed)
    #for batch_idx, tup in tqdm(enumerate(gen)):
    for batch_idx, tup in enumerate(gen):
        with torch.no_grad():

            data, target, hadm_ids, _, descs = tup
            if args.model.find("bert") != -1:
                word, mask, segment, batch_size, chunk_num = data
                data, target = (torch.LongTensor(word), torch.LongTensor(mask), torch.LongTensor(segment), batch_size,
                            chunk_num), torch.FloatTensor(target)
            else:
                if args.use_pos:
                    word, pos = data
                    data, target = (torch.LongTensor(word), torch.LongTensor(pos)), torch.FloatTensor(target)
                else:
                    data, target = torch.LongTensor(data), torch.FloatTensor(target)

            if gpu >= 0:
                if args.model.find("bert") != -1:
                    word, mask, segment, batch_size, chunk_num = data
                    data = (word.cuda(gpu), mask.cuda(gpu), segment.cuda(gpu), batch_size, chunk_num)
                    target = target.cuda(gpu)
                else:
                    if args.use_pos:
                        word, pos = data
                        data = (word.cuda(gpu), pos.cuda(gpu))
                        target = target.cuda(gpu)
                    else:
                        data = data.cuda(gpu)
                        target = target.cuda(gpu)

            model.zero_grad()

            if desc_embed:
                desc_data = descs
            else:
                desc_data = None

            #get an attention sample for 2% of batches
            get_attn = samples and (np.random.rand() < 0.02 or (fold == 'test' and testing))
            output, loss, alpha = model(data, target, desc_data=desc_data, get_attention=get_attn)

            # feili
            # output = F.sigmoid(output)
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()
            # feili
            # losses.append(loss.data[0])
            losses.append(loss.item())
            target_data = target.data.cpu().numpy()
            if get_attn and samples:
                interpret.save_samples(data, output, target_data, alpha, window_size, epoch, tp_file, fp_file, dicts=dicts)

            #save predictions, target, hadm ids
            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)
            hids.extend(hadm_ids)

    #close files if needed
    if samples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
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
                                                    "conv_attn_ldep", 'bert_conv_attn', 'bert', 'resnet_attn'], help="model")
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

