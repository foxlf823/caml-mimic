"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv

from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

from constants import MIMIC_3_DIR

#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

def write_discharge_summaries(out_file):
    notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_file


import nltk
nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')

def my_write_discharge_summaries(out_file, min_sentence_len):
    notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            next(notereader)

            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]

                    all_sents_inds = []
                    generator = nlp_tool.span_tokenize(note)
                    for t in generator:
                        all_sents_inds.append(t)

                    text = ""
                    for ind in range(len(all_sents_inds)):
                        start = all_sents_inds[ind][0]
                        end = all_sents_inds[ind][1]

                        sentence_txt = note[start:end]

                        tokens = [t.lower() for t in tokenizer.tokenize(sentence_txt) if not t.isnumeric()]
                        if ind == 0:
                            text += '[CLS] ' + ' '.join(tokens) + ' [SEP]'
                        else:
                            text += ' [CLS] ' + ' '.join(tokens) + ' [SEP]'

                    text = '"' + text + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')


    return out_file

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word
import re
pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

def prepare_wordemb_corpus(out_file):
    notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    file_num = 0
    sentence_num = 0
    token_num = 0
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))

            notereader = csv.reader(csvfile)
            next(notereader)

            for line in tqdm(notereader):
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    file_num += 1

                    all_sents_inds = []
                    generator = nlp_tool.span_tokenize(note)
                    for t in generator:
                        all_sents_inds.append(t)

                    sentence_num += len(all_sents_inds)
                    for ind in range(len(all_sents_inds)):
                        start = all_sents_inds[ind][0]
                        end = all_sents_inds[ind][1]

                        sentence_txt = note[start:end]
                        tokens = []
                        # not lower, number normed
                        tokens2 = my_tokenize(sentence_txt)
                        for t in tokens2:
                            t = normalize_word(t)
                            tokens.append(t)

                        token_num += len(tokens)
                        outfile.write(' '.join(tokens)+" ")

    print("discharge summary number {}".format(file_num))
    print('sentence number {}'.format(sentence_num))
    print('token number  {}'.format(token_num))

    return

if __name__ == '__main__':
    tokens = my_tokenize("hello word.")
    pass
