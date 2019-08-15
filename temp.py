
import dataproc.extract_wvs as extract_wvs
from constants import *
from dataproc.word_embeddings import fasttext_embeddings

# Y = 'full'
# extract_wvs.gensim_to_embeddings('%s/processed_full.w2v' % '/Users/feili/PycharmProjects/caml-mimic/mimicdata/mimic3',
#                                  '%s/vocab.csv' % '/Users/feili/PycharmProjects/caml-mimic/mimicdata/mimic3', Y)

# fasttext_file = fasttext_embeddings('full', '%s/disch_full.csv' % '/home/lif/caml-mimic/mimicdata/mimic3', 100, 0, 5)

# from dataproc.extract_wvs import gensim_to_fasttext_embeddings
#
# gensim_to_fasttext_embeddings('%s/processed_full.fasttext' % '/Users/feili/PycharmProjects/caml-mimic/mimicdata/mimic3',
#                               '%s/vocab.csv' % '/Users/feili/PycharmProjects/caml-mimic/mimicdata/mimic3', 'full')

import csv
def prepare_elmo_raw_text(filename, outfilename, append=True):
    if append:
        outfile = open(outfilename, 'a')
    else:
        outfile = open(outfilename, 'w')

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        # header
        next(r)

        for row in r:

            text = row[2]

            tokens_ = text.split()
            tokens = []
            for token in tokens_:
                if token == '[CLS]':
                    continue
                if token == '[SEP]': # end of sentence
                    if len(tokens) != 0:
                        for idx, t in enumerate(tokens):
                            if idx == len(tokens)-1:
                                outfile.write(t+'\n')
                            else:
                                outfile.write(t + ' ')
                    tokens = []
                    continue

                tokens.append(token)

    outfile.close()
    return

if __name__ == "__main__":
    prepare_elmo_raw_text('./mimicdata/my_mimic3/train_full.csv', './mimic3_full_elmo.txt')
    prepare_elmo_raw_text('./mimicdata/my_mimic3/dev_full.csv', './mimic3_full_elmo.txt')
    prepare_elmo_raw_text('./mimicdata/my_mimic3/test_full.csv', './mimic3_full_elmo.txt')