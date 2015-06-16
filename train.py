import sys
from glob import glob
from factory.models import *


def main():
    args = sys.argv

    if len(args) < 4:
        print('Please specify the method, input files (comma-separated), and output file.')
        print('One of the following methods work: {}'.format([m for m in globals() if m.startswith('train_')]))
        sys.exit(0)

    paths_ = args[2].split(',')
    out = args[3]

    if len(args) > 4:
        # kwargs should be passed in the form `foo=bar,hey=you`
        kwargs = dict(p.split('=') for p in args[4].split(','))
    else:
        kwargs = {}

    paths = []
    for path in paths_:
        if '*' in path:
            paths += glob(path)
        else:
            paths.append(path)

    # Convenient, but hacky
    globals()[args[1]](paths, out, **kwargs)


if __name__ == '__main__':
    main()