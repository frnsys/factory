from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from nltk.tokenize import sent_tokenize, word_tokenize
from sup.progress import Progress
from factory.knowledge import Bigram


def train_doc2vec(paths, out='data/model.d2v', **kwargs):
    """
    Train a doc2vec model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    print('Training doc2vec model...')
    m = Doc2Vec(_doc2vec_doc_stream(paths, n, sentences=False),
                size=400, window=8, min_count=2, workers=8)

    print('Saving...')
    m.save(out)


def _doc2vec_doc_stream(paths, n, sentences=True):
    """
    Generator to feed sentences to the dov2vec model.
    """
    phrases = Bigram()

    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)

                # We do minimal pre-processing here so the model can learn
                # punctuation
                line = line.lower()

                if sentences:
                    for sent in sent_tokenize(line):
                        tokens = word_tokenize(sent)
                        yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i)])
                else:
                    tokens = word_tokenize(line)
                    yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i)])
