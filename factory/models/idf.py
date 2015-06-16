import json
import math
from collections import defaultdict
from sup.parallel import parallelize
from factory.util import merge, split_file, doc_stream
from factory.knowledge import Bigram


def train_idf(paths, out='data/idf.json', **kwargs):
    """
    Train a IDF model on a list of files (parallelized).
    """
    print('Preparing files...')
    args = []
    method = kwargs.get('method', 'keyword')
    for path in paths:
        args += [(file, method) for file in split_file(path, chunk_size=5000)]

    # Leave a generous timeout in case the
    # phrases model needs to be loaded.
    print('Counting terms...')
    results = parallelize(IDFCounter, args, timeout=360)

    # Serial processing
    #results = []
    #ncount = len(args)
    #p = Progress()
    #obj = IDFCounter()
    #for i, arg in enumerate(args):
        #p.print_progress(i/ncount)
        #results.append(IDFCounter.run(*arg))

    idfs, n_docs = zip(*results)

    print('Merging...')
    idf = merge(idfs)

    print('Computing IDFs...')
    N = sum(n_docs)
    for k, v in idf.items():
        idf[k] = math.log(N/v)
        # v ~= N/(math.e ** idf[k])

    # Keep track of N to update IDFs
    idf['_n_docs'] = N

    with open(out, 'w') as f:
        json.dump(idf, f)


class IDFCounter():
    """
    Using this class for parallel processing so each process
    has its own bigram process connection.
    """
    def __init__(self):
        self.bigram = Bigram()

    def run(self, path, method):
        N = 0
        idf = defaultdict(int)
        for tokens in doc_stream(path, method=method, phrases_model=self.bigram):
            N += 1
            # Don't count freq, just presence
            for token in set(tokens):
                idf[token] += 1
        return idf, N
