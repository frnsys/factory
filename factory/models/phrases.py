from gensim.models import Phrases
from sup.progress import Progress
from nytnlp.clean import strip_punct
from nltk.tokenize import sent_tokenize, word_tokenize


def train_phrases(paths, out='data/bigram_model.phrases', **kwargs):
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
