"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

from constants import *
from pytorch_pretrained_bert import BertTokenizer
import nltk

class Batch:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self, desc_embed, args):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []
        self.use_pos = args.use_pos
        if self.use_pos:
            self.docs_pos = []

    def add_instance(self, instance, ind2c, c2ind, w2ind, dv_dict, num_labels):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        row = instance['row']
        hadm_id = int(row[1])

        length = int(row[4])
        cur_code_set = set()
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []
        #get codes as a multi-hot vector
        for l in row[3].split(';'):
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return
        if self.desc_embed:
            for code in cur_code_set:
                l = ind2c[code]
                if l in dv_dict.keys():
                    #need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[l][:])
                else:
                    desc_vecs.append([len(w2ind)+1])
        #OOV words are given a unique index at end of vocab lookup

        text = instance['word']
        if self.use_pos:
            pos_tags = instance['pos']

        #build instance
        self.docs.append(text)
        if self.use_pos:
            self.docs_pos.append(pos_tags)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        #reset length
        self.length = min(self.max_length, length)

    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        if self.use_pos:
            padded_doc_pos = []
        for i, doc in enumerate(self.docs):
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)

            if self.use_pos:
                doc_pos = self.docs_pos[i]
                if len(doc_pos) < self.length:
                    doc_pos.extend([0] * (self.length - len(doc_pos)))
                padded_doc_pos.append(doc_pos)
        self.docs = padded_docs
        if self.use_pos:
            self.docs_pos = padded_doc_pos

    def to_ret(self):
        if self.use_pos:
            return (np.array(self.docs), np.array(self.docs_pos)), np.array(self.labels), np.array(self.hadm_ids), self.code_set, \
                   np.array(self.descs)
        else:
            return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
                   np.array(self.descs)

class Batch_bert:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self, desc_embed, tokenizer, chunk_len):
        self.docs = []
        self.docs_mask = []
        self.docs_segment = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len

    def add_instance(self, row, ind2c, c2ind, w2ind, dv_dict, num_labels):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        labels = set()
        hadm_id = int(row[1])
        text = row[2]

        cur_code_set = set()
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []
        #get codes as a multi-hot vector
        for l in row[3].split(';'):
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return
        if self.desc_embed:
            for code in cur_code_set:
                l = ind2c[code]
                if l in dv_dict.keys():
                    #need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[l][:])
                else:
                    desc_vecs.append([len(w2ind)+1])

        text = self.tokenizer.tokenize(text)
        length = len(text)


        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]


        #build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        # the doc is ordered from short to long, so self.length records the max length in this batch
        self.length = min(self.max_length, length)

    def pad_docs(self):
        self.batch_size = len(self.docs)

        if self.length == self.chunk_len:
            self.chunk_num = 1
        else:
            self.chunk_num = self.length // self.chunk_len + 1

        padded_docs = []
        padded_docs_mask = []
        padded_docs_segment = []
        for doc in self.docs:
            # split the doc into chunks
            for chunk_idx in range(self.chunk_num):
                start = chunk_idx * self.chunk_len
                end = chunk_idx * self.chunk_len + self.chunk_len
                if start >= len(doc): # make a empty chunk
                    chunk = ['[PAD]'] * self.chunk_len
                elif start < len(doc) and end > len(doc):
                    chunk = doc[start:]
                else:
                    chunk = doc[start:end]

                chunk.insert(0, '[CLS]')
                chunk.append('[SEP]')
                #print(chunk)
                chunk = self.tokenizer.convert_tokens_to_ids(chunk)

                actual_chunk_length = len(chunk)
                mask = [1] * actual_chunk_length
                segment = [0] * actual_chunk_length

                expected_chunk_length = self.chunk_len+2 # [CLS]+[SEP]

                if actual_chunk_length < expected_chunk_length:
                    # pad the chunk into chunk_len
                    chunk.extend([0] * (expected_chunk_length - actual_chunk_length))
                    mask.extend([0] * (expected_chunk_length - actual_chunk_length))
                    segment.extend([0] * (expected_chunk_length - actual_chunk_length))

                padded_docs.append(chunk)
                padded_docs_mask.append(mask)
                padded_docs_segment.append(segment)


        self.docs = padded_docs
        self.docs_mask = padded_docs_mask
        self.docs_segment = padded_docs_segment

    def to_ret(self):
        return (np.array(self.docs), np.array(self.docs_mask), np.array(self.docs_segment), self.batch_size, self.chunk_num), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
               np.array(self.descs)

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def data_generator(instances, args, tokenizer, filename, dicts, batch_size, num_labels, desc_embed=False, version='mimic3'):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
            version: which (MIMIC) dataset
        Yields:
            np arrays with data for training loop.
    """
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']

    if args.model.find("bert") != -1:
        cur_inst = Batch_bert(desc_embed, tokenizer, args.bert_chunk_len)
    else:
        cur_inst = Batch(desc_embed, args)
    for instance in instances:
        #find the next `batch_size` instances
        if len(cur_inst.docs) == batch_size:
            cur_inst.pad_docs()
            yield cur_inst.to_ret()
            #clear
            if args.model.find("bert") != -1:
                cur_inst = Batch_bert(desc_embed, tokenizer, args.bert_chunk_len)
            else:
                cur_inst = Batch(desc_embed, args)
        cur_inst.add_instance(instance, ind2c, c2ind, w2ind, dv_dict, num_labels)
    cur_inst.pad_docs()
    yield cur_inst.to_ret()

def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    # if args.model.find('bert') != -1:
    #     tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    #     with open(vocab_file, 'r') as vocabfile:
    #         for i, line in enumerate(vocabfile):
    #             line = line.rstrip()
    #             if line != '':
    #                 word_pieces = tokenizer.tokenize(line.strip())
    #                 for word_piece in word_pieces:
    #                     vocab.add(word_piece)
    #
    #         ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    #         w2ind = {w: i for i, w in ind2w.items()}
    # else:
    #     with open(vocab_file, 'r') as vocabfile:
    #         for i,line in enumerate(vocabfile):
    #             line = line.rstrip()
    #             if line != '':
    #                 vocab.add(line.strip())
    #     #hack because the vocabs were created differently for these models
    #     if args.public_model and args.Y == 'full' and args.version == "mimic3" and args.model == 'conv_attn':
    #         ind2w = {i:w for i,w in enumerate(sorted(vocab))}
    #     else:
    #         ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    #     w2ind = {w:i for i,w in ind2w.items()}

    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            # if line.strip() in vocab:
            #     print(line)
            if line != '':
                vocab.add(line.strip())
    # hack because the vocabs were created differently for these models
    if args.public_model and args.Y == 'full' and args.version == "mimic3" and args.model == 'conv_attn':
        ind2w = {i: w for i, w in enumerate(sorted(vocab))}
    else:
        ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}

    return ind2w, w2ind

def load_lookups(args, desc_embed=False):
    """
        Inputs:
            args: Input arguments
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(args, args.vocab)

    #get code and description lookups
    if args.Y == 'full':
        ind2c, desc_dict = load_full_codes(args.data_path, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        desc_dict = load_code_descriptions()
    c2ind = {c:i for i,c in ind2c.items()}

    #get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(args.Y, version=args.version)
    else:
        dv_dict = None

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict}

    if args.model.find('lco') != -1:
        label_matrix = build_concurr_matrix(args, c2ind, ind2c)
        dicts['label_matrix'] = label_matrix

    return dicts

def build_concurr_matrix(args, c2ind, ind2c):

    if args.version == 'mimic3':
        label_matrix = np.zeros((len(c2ind), len(c2ind)), dtype=np.float32)

        with open(args.data_path, 'r') as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                codes = row[3].split(';')
                for i, code_i in enumerate(codes):

                    j = 0
                    while j < i:
                        code_j = codes[j]
                        code_i_idx = c2ind[code_i]
                        code_j_idx = c2ind[code_j]
                        label_matrix[code_i_idx,code_j_idx] = label_matrix[code_i_idx,code_j_idx] + 1
                        # symmetric
                        label_matrix[code_j_idx, code_i_idx] = label_matrix[code_j_idx, code_i_idx] + 1

                        j += 1

        label_matrix += np.identity(label_matrix.shape[0])

    else:
        raise RuntimeError("not support {}".format(args.version))



    return label_matrix

def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    #get description lookup
    # desc_dict = load_code_descriptions(version=version)
    desc_dict = None
    #build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c, desc_dict

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(version='mimic3'):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for i,row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def load_description_vectors(Y, version='mimic3'):
    #load description one-hot vectors from file
    dv_dict = {}
    if version == 'mimic2':
        data_dir = MIMIC_2_DIR
    else:
        data_dir = MIMIC_3_DIR
    with open("%s/description_vectors.vocab" % (data_dir), 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict
