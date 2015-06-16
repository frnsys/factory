import json
from collections import defaultdict
from sup.parallel import parallelize
from factory.util import merge, split_file, doc_stream
from factory.knowledge import Bigram


def train_tf(paths, out='data/tf.json', **kwargs):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    print('Preparing files')
    args = []
    method = kwargs.get('method', 'keyword')
    for path in paths:
        args += [(file, method) for file in split_file(path, chunk_size=5000)]

    # Leave a generous timeout in case the
    # phrases model needs to be loaded.
    print('Counting terms...')
    results = parallelize(TFCounter, args, timeout=360)

    print('Merging...')
    tf = merge(results)

    with open(out, 'w') as f:
        json.dump(tf, f)


class TFCounter():
    """
    Using this class for parallel processing so each process
    has its own bigram process connection.
    """
    def __init__(self):
        self.bigram = Bigram()

    def run(self, path, method):
        """
        Count term frequencies for a single file.
        """
        tf = defaultdict(int)
        for tokens in doc_stream(path, method=method, phrases_model=self.bigram):
            for token in tokens:
                tf[token] += 1
        return tf
