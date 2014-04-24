#!/urs/local/bin/python
"""Bayesian CRP blocked sampling segmenter.

Usage:
    seg.py <file_to_segment.ylt>

Options:
    -h --help     Show this screen.
    --version     Show version.

"""

from docopt import docopt
from collections import defaultdict
import numpy as np
import random, time

DEBUG = True
MAX_LEN_SENTENCE = 512
#integers = np.arange(MAX_LEN_SENTENCE)


class DummyCRP:
    def predProb(self, _):
        return 1.


class CRP:
    def __init__(self, aboveLevel=None, alpha=1.):
        self._alpha = alpha * 1.0  # just to be sure it's a float
        self._tables = defaultdict(lambda: 0)  # number of customers at table
        self._sum = self._alpha  # includes alpha
        self._aboveLevel = DummyCRP()
        if aboveLevel != None:
            assert hasattr(aboveLevel, "predProb")
            self._aboveLevel = aboveLevel

    def addCustomer(self, table):
        self._tables[table] += 1
        self._sum += 1

    def removeCustomer(self, table):
        "Returns if the table is empty or not."
        if table not in self._tables:
            return True
        self._tables[table] -= 1
        self._sum -= 1
        if self._tables[table] <= 0:
            self._tables.pop(table)
            return True
        return False

    def prob(self, table):
        # TODO check
        return (self._tables[table] + self._alpha
                * self._baseProb(table)) / self._sum 

    def _new_table_prob(self):
        return self._alpha / self._sum

    def _baseProb(self, table):
        return self._aboveLevel.predProb(table)

    def predProb(self, table):
        """Computes the predictive probability of siting a customer at "table":
        P(customer@table | tables) = (counts(customers@table) + alpha
                                      * baseProb(customer@table) / (alpha + n)
        """
        if table in self._tables:
            return self.prob(table)
        else:
            return self._new_table_prob()

    def verify(self):
        "Checks that everytime something is added or removed, sum is updated."
        #print self._tables
        print "self sum:", self._sum
        print "tables values sum:", np.sum(self._tables.values())
        print "alpha:", self._alpha
        assert(self._sum == np.sum(self._tables.values()) + self._alpha)


def search_before_after(line, j, k):
    """Search for "k"-level boundaries before and after "j" in "line". """
    # TODO memoize (by removing line for "i" maybe)
    before = 0
    for cur in xrange(j-1, -1, -1):
        if line[cur] >= k:
            before = cur + 1
            break
    after = len(line) + 1
    for cur in xrange(j+1, len(line)):
        if line[cur] >= k:
            after = cur + 1
            break
    return before, after



def sample_and_segment(data, nlvls=2, niter=100):
    """Samples for "niter" with "nlvls" or CRP and segments data.

    We initialize by assuming that each line is starter by a "nlvls" boundary

    We update by considering (in this order):
    Variant 1:
    for each level L (in ascending order)
        for each boundary B
            if B > 0:
                remove B by setting B=0 and updating the CRPs accordingly
            sample a boundary B=L here prop. to the joint on CRPs   [TODO]

    Variant 2 (that we are doing):
    for each level L (in ascending order)
        for each boundary B
            if B < L-1:
                continue  # we don't sample if there is no lower-level boundary
            if B > L-1:
                remove B by setting B=L-1 and updating CRPs[>=L] accordingly
            sample a boundary B=L here prop. to joint CRP[>=L]

    Examples with 2 levels (L in {1, 2}),
    init:
        y u w a n t t u s i D 6 b U k
     (2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (2)

    update at lvl L=1 for the boundary in brackets:
      y   u w a n t t u s i D 6 b U k
       [0] 0 0 0 0 0 0 0 0 0 0 0 0 0
    iterating:
      y   u w a n t t u s i D 6 b U k
       [0] 0 0 0 0 0 0 0 0 0 0 0 0 0 (prob = 1./2)
      y u   w a n t t u s i D 6 b U k
       0 [1] 0 0 0 0 0 0 0 0 0 0 0 0 (prob = 1./2)
      y u w   a n t t u s i D 6 b U k
       0 1 [0] 0 0 0 0 0 0 0 0 0 0 0 (prob = 1./3)
      y u w a   n t t u s i D 6 b U k
       0 1 0 [0] 0 0 0 0 0 0 0 0 0 0 (prob = 1./3)
    TODO
    """

    crps = []
    for i in xrange(nlvls):
        if i == 0:
            crps.append(CRP())
        else:
            crps.append(CRP(crps[i-1]))
    crps.reverse()
    crps = [None] + crps # concat [None] at head for easier indices alignment
    segmentation = [np.zeros(len(l)-1, dtype='int32') for l in data]
    ### This part is sample
    for iteration_number in xrange(niter):
        print "starting iteration:", iteration_number
        start_t = time.time()
        for i, line in enumerate(segmentation):
            for lvl in xrange(1, nlvls+1):
                lower_lvl = lvl - 1
                for j, b_lvl in enumerate(line):
                ###for j, b_lvl in enumerate(segmentation[i]):
                    ###line = segmentation[i]
                    if b_lvl < lower_lvl:
                        continue
                    if b_lvl >= lvl:
                        # remove the boundary by setting it at lower_lvl
                        line[j] = lower_lvl
                        for k in xrange(lvl, nlvls+1):
                            # find the segments/customers that are affected
                            before, after = search_before_after(line, j, k)
                            # remove these customers
                            crps[k].removeCustomer(''.join(data[i][before:j+1]))
                            crps[k].removeCustomer(''.join(data[i][j+1:after]))
                    before, after = search_before_after(line, j, lvl)
                    oneword = ''.join(data[i][before:after])
                    twowords = (''.join(data[i][before:j+1]),
                            ''.join(data[i][j+1:after]))
                    without_b = crps[lvl].predProb(oneword)
                    with_b = crps[lvl].predProb(twowords[0]) * \
                            crps[lvl].predProb(twowords[1])
                    if random.random() > with_b/(without_b + with_b):
                        line[j] = lvl
                        crps[lvl].addCustomer(twowords[0])
                        crps[lvl].addCustomer(twowords[1])
                    else:
                        crps[lvl].addCustomer(oneword)
        if DEBUG:
            for lvl, crp in enumerate(crps):
                if crp != None:
                    print "table at lvl:", lvl
                    crp.verify()
        print "took:", time.time() - start_t, "seconds"
    for lvl, crp in enumerate(crps):
        if crp != None:
            print "table at lvl:", lvl
            print crp._tables

    ### This part is segment




if __name__ == '__main__':
    arguments = docopt(__doc__, version='segmampler version 0.1')
    dataset = []
    with open(arguments['<file_to_segment.ylt>']) as f:
        dataset = map(lambda s: s.rstrip('\n').split(), f.readlines())
    sample_and_segment(dataset)


