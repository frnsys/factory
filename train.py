import os
import sys
import json
import math
from glob import glob
from itertools import chain, islice
from collections import Counter, defaultdict
from gensim.models import Phrases
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from nltk.tokenize import word_tokenize, sent_tokenize
from sup.progress import Progress
from sup.parallel import parallelize
from nytnlp.clean import strip_punct
from nytnlp.tokenize.keyword import keyword_tokenize


def train_doc2vec(paths, out='data/model.d2v'):
    """
    Train a doc2vec model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    print('Training doc2vec model...')
    m = Doc2Vec(_doc2vec_doc_stream(paths, n),
                size=400, window=8, min_count=2, workers=8)

    print('Saving...')
    m.save(out)


def train_phrases(paths, out='data/bigram_model.phrases'):
    """
    Train a bigram phrase model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    # Change to use less memory. Default is 40m.
    max_vocab_size = 40000000

    print('Training bigrams...')
    bigram = Phrases(_phrase_doc_stream(paths, n), max_vocab_size=max_vocab_size, threshold=8.)

    print('Saving...')
    bigram.save(out)

    print('Some examples:')
    docs = [
        ['the', 'new', 'york', 'times', 'is', 'a', 'newspaper'],
        ['concern', 'is', 'rising', 'in', 'many', 'quarters', 'that', 'the', 'united', 'states', 'is', 'retreating', 'from', 'global', 'economic', 'leadership', 'just', 'when', 'it', 'is', 'needed', 'most'],
        ['the', 'afghan', 'president', 'ashraf', 'ghani', 'blamed', 'the', 'islamic', 'state', 'group'],
        ['building', 'maintenance', 'by', 'the', 'hrynenko', 'family', 'which', 'owns', 'properties', 'in', 'the', 'east', 'village'],
        ['a', 'telegram', 'from', 'the', 'american', 'embassy', 'in', 'constantinople', 'to', 'the', 'state', 'department', 'in', 'washington']
    ]
    for r in bigram[docs]:
        print(r)


def train_idf(paths, out='data/idf.json'):
    """
    Train a IDF model on a list of files (parallelized).
    """
    for path in paths:
        args = [(file,) for file in _split_file(path, chunk_size=5000)]

    results = parallelize(_count_idf, args)
    idfs, n_docs = zip(*results)

    print('Merging...')
    idf = _merge(idfs)
    N = sum(n_docs)

    for k, v in idf.items():
        idf[k] = math.log(N/v)

    with open(out, 'w') as f:
        json.dump(idf, f)


def _count_idf(path):
    """
    Count term frequencies and documents for a single file.
    """
    N = 0
    idf = defaultdict(int)
    for tokens in _idf_doc_stream(path):
        N += 1
        # Don't count freq, just presence
        for token in set(tokens):
            idf[token] += 1
    return idf, N


def train_tf(paths, out='data/tf.json'):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    for path in paths:
        args = [(file,) for file in _split_file(path, chunk_size=5000)]

    results = parallelize(_count_tf, args)

    print('Merging...')
    tf = _merge(results)

    with open(out, 'w') as f:
        json.dump(tf, f)


def _count_tf(path):
    """
    Count term frequencies for a single file.
    """
    tf = defaultdict(int)
    for tokens in _tf_doc_stream(path):
        for token in tokens:
            tf[token] += 1
    return tf


def _phrase_doc_stream(paths, n):
    """
    Generator to feed sentences to the phrase model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line.lower()):
                    tokens = word_tokenize(strip_punct(sent))
                    yield tokens


def _doc2vec_doc_stream(paths, n):
    """
    Generator to feed sentences to the dov2vec model.
    """
    phrases = Phrases.load('data/bigram_model.phrases')

    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)

                # We do minimal pre-processing here so the model can learn
                # punctuation
                for sent in sent_tokenize(line.lower()):
                    tokens = word_tokenize(sent)
                    yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i)])


def _idf_doc_stream(path):
    """
    Generator to feed sentences to IDF trainer.
    """
    with open(path, 'r') as f:
        for line in f:
            yield keyword_tokenize(line)
            #for sent in sent_tokenize(line):
                #yield keyword_tokenize(sent)


def _tf_doc_stream(path):
    """
    Generator to feed sentences to TF trainer.
    """
    with open(path, 'r') as f:
        for line in f:
            #yield keyword_tokenize(line)
            yield word_tokenize(line.lower())


def _merge(dicts):
    """
    Merges a list of dicts, summing their values.
    """
    merged = sum([Counter(d) for d in dicts], Counter())
    return dict(merged)


def _chunks(iterable, n):
    """
    Splits an iterable into chunks of size n.
    """
    iterable = iter(iterable)
    while True:
        # store one line in memory,
        # chain it to an iterator on the rest of the chunk
        yield chain([next(iterable)], islice(iterable, n-1))


def _split_file(path, chunk_size=50000):
    """
    Splits the specified file into smaller files.
    """
    with open(path) as f:
        for i, lines in enumerate(_chunks(f, chunk_size)):
            file_split = '{}.{}'.format(os.path.basename(path), i)
            chunk_path = os.path.join('/tmp', file_split)
            with open(chunk_path, 'w') as f:
                f.writelines(lines)
            yield chunk_path


if __name__ == '__main__':
    args = sys.argv

    if len(args) < 4:
        print('Please specify the method, input files (comma-separated), and output file.')
        print('One of the following methods work: {}'.format([m.replace('train_', '') for m in globals() if m.startswith('train_')]))
        sys.exit(0)

    paths_ = args[2].split(',')
    out = args[3]

    paths = []
    for path in paths_:
        if '*' in path:
            paths += glob(path)
        else:
            paths.append(path)

    # Convenient, but hacky
    globals()['train_{}'.format(args[1])](paths, out)