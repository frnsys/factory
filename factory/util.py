import os
import numpy as np
from itertools import chain, islice
from collections import defaultdict
from sup.parallel import parallelize
from nytnlp.common import spacy
from nltk.tokenize import word_tokenize
from nytnlp.tokenize.keyword import keyword_tokenize


def merge(dicts):
    """
    Merges a list of dicts, summing their values.
    (Parallelized wrapper around `_count`)
    """
    # Split into 20 chunks
    args_chunks = [(args,) for args in np.array_split(dicts, 20)]
    results = parallelize(_count, args_chunks)
    return _count(results)


def _count(dicts):
    """
    Merge a list of dicts, summing their values.
    """
    counts = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            counts[k] += v
    return counts


def _chunks(iterable, n):
    """
    Splits an iterable into chunks of size n.
    """
    iterable = iter(iterable)
    while True:
        # store one line in memory,
        # chain it to an iterator on the rest of the chunk
        yield chain([next(iterable)], islice(iterable, n-1))


def split_file(path, chunk_size=50000):
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


def doc_stream(path, method='word', phrases_model=None):
    """
    Generator to feed tokenized documents (treating each line as a document).
    """
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                if method == 'entity':
                    res = spacy(line, entity=True, parse=False, tag=True)
                    yield [e.string.lower() for e in res.ents]
                if method == 'keyword':
                    yield keyword_tokenize(line, phrases_model=phrases_model)
                else:
                    yield word_tokenize(line)
