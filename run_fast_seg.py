"""Fast version of seg.py.

Usage:
    run_fast_seg.py <file_to_segment.ylt> [--iters=niterations]

Options:
    -h --help     Show this screen.
    --version     Show version.

"""

#from fast_seg import sample_and_segment
#from fast_seg_colloc import sample_and_segment
from seg_colloc_2 import sample_and_segment
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Cython segmampler version 0.1')
    dataset = []
    fname = arguments['<file_to_segment.ylt>']
    niter = 10
    if arguments['--iters'] != None:
        niter = int(arguments['--iters'])
    with open(fname) as f:
        dataset = map(lambda s: s.rstrip('\n').split(), f.readlines())
    segmented = sample_and_segment(dataset, niter=niter)
    for level, seg in enumerate(segmented):
        with open(fname.split('.')[0] + '_' + str(level) + '.seg', 'w') as wf:
            wf.write('\n'.join(map(lambda l: ' '.join(l), seg)))
