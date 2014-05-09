#!/usr/local/bin/python
"""Bayesian CRP break-point sampling segmenter.

Usage:
    seg.py <file_to_segment.ylt> [--iters=niterations]

Options:
    -h --help     Show this screen.
    --version     Show version.

implements the collocation model

observations at the colloc-crp level are ( w,...,w ), i.e. tuples of (potentially) multiple words
observations at the word-crp level are (w), i.e. singleton tuples

"""

from docopt import docopt
from collections import defaultdict
from math import log
import numpy as np
import random, time
from scipy.special import gammaln

DEBUG = False
MAX_LEN_SENTENCE = 512
NSEGS = 12 #for the test corpus

BURNIN = 1000
EVERY = 10

#integers = np.arange(MAX_LEN_SENTENCE)


class DummyCRP:
    def predProb(self, _):
        return 1.


class CRP:
    """
      observations are tuples -- for the uniCRP, singleton tuples, otherwise, tuples of tuples (of tuples)

      if aboveLevel is another CRP, the basedistribution generates sequences of observations

      for example, the colloc-distribution backs off to a product of unigram probabilities, that is,
      if table= ((the,),(cat,)), we back off to
      uni.predProb((the)) * 0.5 * uni.predProb((cat))

      if no above-level, we back off to a character process, that is, uni backs off to
      (0.5/#phons)**len(table)

      if a new table is opened and aboveLevel is not None, we need to send a customer for each element of the tuple
      if table= ((the,),(cat,)) in colloc, we add (the,) and (cat,) to the uni-CRP
    """

    def __init__(self, aboveLevel=None, alpha=1.0):
        self._alpha = alpha * 1.0  # just to be sure it's a float
        self._tables = defaultdict(lambda: 0)  # number of customers at table
        self._sum = self._alpha  # includes alpha
        self._aboveLevel = aboveLevel #DummyCRP()
        if self._aboveLevel == None:
            self._genSegs = 0 #base-distribution

    def baseProb(self,table):
        if self._aboveLevel is None:
            return (0.5/NSEGS)**len(table[0])
        else:
            ### correct
            # res = 1.0
            # for w in table:
            #     res = res * 0.5 * self._aboveLevel.predProb(w)
            #     self._aboveLevel.addCustomer(w)
            # for w in table:
            #     self._aboveLevel.removeCustomer(w)                
            # return res
            ### hacky
            return reduce(lambda x,y: x*y, (0.5*self._aboveLevel.predProb(w) for w in table))

    def logProb(self):
        res = gammaln(self._alpha) - gammaln(self._sum)
        res += reduce(lambda x,y:x+y,[x for x in self._tables.itervalues()])
        res += len(self._tables)*log(self._alpha)
        if self._aboveLevel!=None:
            res += self._aboveLevel.logProb()
        else:
            res += self._genSegs * log(0.5/NSEGS)
        return res
        

    def __str__(self):
        res = []
        for (k,c) in self._tables.iteritems():
            res.append("(%s, %d)"%(k, c))
        return ", ".join(res)

    def addCustomer(self, table):
        #print "add",table
        if self._tables[table]==0:
            if self._aboveLevel != None:
                for w in table:
                    self._aboveLevel.addCustomer(w)
            else:
                self._genSegs += len(table[0])
        self._tables[table] += 1
        self._sum += 1

    def removeCustomer(self, table):
        "Returns if the table is empty after removal or not."
        #print "remove",table
        if table not in self._tables:
            raise Exception("Can't remove %s from %s"%(table,self._tables))
        self._tables[table] -= 1
        self._sum -= 1
        if self._tables[table] == 0:
            self._tables.pop(table)
            if self._aboveLevel != None:
                for w in table:
                    self._aboveLevel.removeCustomer(w)
            else:
                self._genSegs -= len(table[0])
            return True
        return False

    def prob(self, table):
        "probability of joining old table"
        # TODO check
        return (self._tables[table] / self._sum)
        #return (self._tables[table] + self._alpha
        #        * self._baseProb(table)) / self._sum 

    def _new_table_prob(self,table):
        "probability of opening new table"
        return self._alpha*self.baseProb(table) / self._sum

    def predProb(self, table):
        """Computes the predictive probability of siting a customer at "table":
        P(customer@table | tables) = (counts(customers@table) + alpha
                                      * baseProb(customer@table)) / (alpha + n)
        """
        if table in self._tables:
            return self.prob(table)
        else:
            #return self._new_table_prob() * self._baseProb(table)
            # TODO check
            #return self._alpha * self._baseProb(table)
            return self._new_table_prob(table)

    def verify(self):
        "Checks that everytime something is added or removed, sum is updated."
        #print self._tables
        print "self sum:", self._sum
        print "tables values sum:", np.sum(self._tables.values())
        print "alpha:", self._alpha
        assert(self._sum == np.sum(self._tables.values()) + self._alpha)



def find(line,utt,start,end,level,rec=True):
    """
      recursively identify a level-collocation that spans start-end
      if rec=False, assumes that there is exactly one level-object between start and end
    """
    #print line,utt,start,end,level,rec
    res = []
    sPos = start
    nPos = sPos+1
    while nPos<end:
        if line[nPos]>=level:
            if level==1:
                res.append(tuple([''.join(utt[sPos:nPos])]))
            else:
                res.append(find(line,utt,sPos,nPos,level-1))
            sPos=nPos
        nPos+=1
    if level==1:
        res.append(tuple([''.join(utt[sPos:nPos])]))
    else:
        res.append(find(line,utt,sPos,nPos,level-1))            
    if rec:
        return tuple(res)
    else:
        return res[0]

def search_before_after(line, j, k):
    """Search for "k"-level boundaries before and after "j" in "line".
       each boundary vector is prefixed and terminated by an additional highest-level boundary
       thus, we index boundaries starting from 1
    """
    # TODO memoize (by removing line for "i" maybe)
    before = j-1
    while line[before]<k:
        before -= 1
    after = j+1
    while line[after]<k:
        after += 1
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
    samples = []
    crps = []
    for i in xrange(nlvls):
        if i == 0:
            crps.append(CRP())
        else:
            crps.append(CRP(aboveLevel=crps[i-1]))
    crps = [None] + crps # concat [None] at head for easier indices alignment
    segmentation = [np.zeros(len(l)+1, dtype='int32') for l in data]
    for s in segmentation:
        s[0]=nlvls
        s[-1]=nlvls
    ### Initialize
    for (line,utt) in zip(segmentation,data):
        for c in find(line,utt,0,len(line)-1,nlvls):
            crps[nlvls].addCustomer(c)
    ### This part is sample
    for iteration_number in xrange(niter):
        print "starting iteration:", iteration_number,"lp=",crps[-1].logProb()
        start_t = time.time()
        for i, line in enumerate(segmentation):
            for j, b_lvl in enumerate(line):
                if j==0 or j==len(line)-1: #skip beginning and end
                    continue
                # find the segments/customers that are affected
                # remove a top-level constitutent that is affected
                # if the boundary is a top-level boundary, two top-level constituents
                # need to be removed
                before_glob = 0
                aft_glob = 0
                before_glob, aft_glob = search_before_after(line, j, nlvls)
                #print "b=",before,"after=",after
                # remove these customers
                if line[j]==nlvls:
                    crps[nlvls].removeCustomer(find(line,data[i],before_glob,j,nlvls,False))
                    crps[nlvls].removeCustomer(find(line,data[i],j,aft_glob,nlvls,False))
                else:
                    crps[nlvls].removeCustomer(find(line,data[i],before_glob,aft_glob,nlvls,False))
                ### calculate all hypotheses
                hypotheses = []
                ### just single stretches
                for plvl in xrange(0,nlvls): # collect all other hypotheses up to nlvls
                    line[j]=plvl
                    oneword = find(line,data[i],before_glob,aft_glob,nlvls,False)
                    p = crps[nlvls].predProb(oneword) 
                    hypotheses.append((plvl,oneword,p))
                ### two top-level constituents
                twowords = (find(line,data[i],before_glob,j,nlvls,False),
                            find(line,data[i],j,aft_glob,nlvls,False))
                p = crps[nlvls].predProb(twowords[0])
                crps[nlvls].addCustomer(twowords[0])
                p = p*crps[nlvls].predProb(twowords[1])
                crps[nlvls].removeCustomer(twowords[0])
                hypotheses.append((nlvls,twowords,p))
                ### sample
                flip = random.random()
                norm = sum(x[2] for x in hypotheses)
                cur = 0
                ti = 0
                while cur<=flip:
                    #print hypotheses[ti],
                    cur += hypotheses[ti][2]/norm
                    #print cur
                    ti += 1
                sampled = hypotheses[ti-1]
                line[j] = sampled[0]
                if sampled[0]==nlvls:
                    crps[nlvls].addCustomer(sampled[1][0])
                    crps[nlvls].addCustomer(sampled[1][1])
                else:
                    crps[nlvls].addCustomer(sampled[1])
        if DEBUG:
            for lvl, crp in enumerate(crps):
                if crp != None:
                    print "table at lvl:", lvl
                    crp.verify()
        if iteration_number+1>BURNIN and iteration_number % EVERY == 0:
            samples.append(segmentation)
        print "took:", time.time() - start_t, "seconds"
    for lvl, crp in enumerate(crps):
        if crp != None:
            print "table at lvl:", lvl
            #print filter(lambda (k,v): v>100, crp._tables.iteritems())
            print crp

    ### This part is segment
    # using just the last sample
    # TODO: max marginal segmentation
    mms = [] 
    for i in xrange(len(data)):
        counts = defaultdict(int)
        for sample in samples:
            counts[tuple(sample[i])]+=1
        mms.append(max(counts.items(),key=lambda x:x[1])[0])
    ret = []
    for lvl in xrange(1, nlvls+1):
        corpus = []
        for i, line in enumerate(mms):
            prev_b = 0
            sentence = []
            for j, b_lvl in enumerate(line[1:-1]):
                if b_lvl >= lvl:
                    sentence.append(''.join(data[i][prev_b:j+1]))
                    prev_b = j+1
            sentence.append(''.join(data[i][prev_b:]))
            corpus.append(sentence)
        ret.append(corpus)
    return ret


if __name__ == '__main__':
    arguments = docopt(__doc__, version='segmampler version 0.1')
    dataset = []
    fname = arguments['<file_to_segment.ylt>']
    niter = 2000
    if arguments['--iters'] != None:
        niter = int(arguments['--iters'])
    with open(fname) as f:
        dataset = map(lambda s: s.rstrip('\n').split(), f.readlines())
    segmented = sample_and_segment(dataset, niter=niter,nlvls=2)
    for level, seg in enumerate(segmented):
        with open(fname.split('.')[0] + '_' + str(level) + '.seg', 'w') as wf:
            wf.write('\n'.join(map(lambda l: ' '.join(l), seg)))

