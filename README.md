# factory

Script to train various NLP models for use in other projects

I often have to train NLP models for various projects in a redundant way - `factory` keeps that training in one place so I don't have to repeat myself

## Setup

    $ pip install -r requirements.txt

## Usage

    $ python train.py <model> <comma-separated input files or glob wildcard pattern>

For example:

    $ python train.py doc2vec file1.txt,file2.txt,file3.txt
    $ python train.py doc2vec corpus/*.txt