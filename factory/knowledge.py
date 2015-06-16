"""
These data sources take extremely long to load,
so this is an abstraction which allows the running
of them in a separate process so they don't constantly
need to be reloaded.

Calling out to a separate process is slower, but
cuts down on loading time, and in parallelization, reduces memory usage.
"""

from gensim.models import Phrases
from multiprocessing.connection import Client

_phrases = None

class Bigram():
    def __getitem__(self, word):
        global _phrases

        # If a phrases model is already loaded, just use that
        if _phrases is not None:
            self.conn = None

        # Otherwise, try to connect to the separate process.
        # Fall back to loading the phrase model here
        elif not hasattr(self, 'conn'):
            try:
                print('Connecting to phrases process...')
                address = ('localhost', 6001)
                self.conn = Client(address, authkey=b'password')
                print('Done connecting to phrases')

            except ConnectionRefusedError:
                self.conn = None
                print('Could not connect to phrases process,')
                print('Loading phrases model...')
                _phrases = Phrases.load('data/bigram_model.phrases')
                print('Done loading phrases')

        if self.conn is not None:
            self.conn.send(word)
            return self.conn.recv()
        else:
            return _phrases[word]
