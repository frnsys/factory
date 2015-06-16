# Factory

Build various NLP models in a parallel manner


## Usage

For more efficient memory-usage in parallel processing, run the bigram phrase model as a separate process:

    $ python service.py phrases

The factory will automatically connect to this process if it is needed and available. Otherwise, it falls back to loading the phrases model directly.


Then run the command you want, for example:

    $ python train.py train_idf "data/*.txt" data/idf.json method=word

You can see all available commands by running:

    $ python train.py

The structure of the commands are generally:

    $ python train.py <command> <path to input> <path to ouput> <kwargs>
