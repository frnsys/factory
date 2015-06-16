"""
Run some slow-loading things as separate process
that way they don't need to keep being reloaded.
"""

import sys
from sup.service import Service
from gensim.models import Phrases


class PhraseService(Service):
    def __init__(self):
        super().__init__(multithreaded=False)

        print('Loading phrases model...')
        self.bigram = Phrases.load('data/bigram_model.phrases')

    def handle(self, conn):
        msg = conn.recv()
        resp = self.bigram[msg]
        conn.send(resp)


def phrases():
    p = PhraseService()
    p.run()


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()
