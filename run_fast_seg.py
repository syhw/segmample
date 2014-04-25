"""Fast version of seg.py.

Usage:
    run_fast_seg.py <file_to_segment.ylt>

Options:
    -h --help     Show this screen.
    --version     Show version.

"""

from fast_seg import sample_and_segment
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__, version='segmampler version 0.1')
    dataset = []
    fname = arguments['<file_to_segment.ylt>']
    with open(fname) as f:
        dataset = map(lambda s: s.rstrip('\n').split(), f.readlines())
    sample_and_segment(dataset)
