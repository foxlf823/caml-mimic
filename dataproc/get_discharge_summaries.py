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

