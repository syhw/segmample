#cython: profile=False
# set profile to True to see where we are spending time / what is Cythonized
#cython: boundscheck=False
# set boundscheck to True when debugging out of bounds errors
#cython: wraparound=False
#cython: cdivision=True
#cython: overflowcheck=False
#cython: infertypes=False
"""Fast version of seg.py functions, to be used as a module.

To be used as a module
"""

import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DTYPE_t
ctypedef np.float64_t CTYPE_t
ctypedef np.intp_t IND_t 
ctypedef np.int32_t INT32_t 
DTYPE = np.float64
import time 
from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

from seg import DummyCRP, CRP

cdef struct Pair:
    IND_t before
    IND_t after


cdef Pair search_before_after(INT32_t[:] line, int j, int k):
    """Search for "k"-level boundaries before and after "j" in "line". """
    cdef Pair ret
    ret.before = 0
    ret.after = 0
    cdef IND_t cur
    cdef IND_t max_length = line.shape[0]
    for cur in range(j-1, -1, -1):
        if line[cur] >= k:
            ret.before = cur + 1
            break
    for cur in range(j+1, max_length):
        if line[cur] >= k:
            ret.after = cur + 1
            break
    return ret


cpdef sample_and_segment(data, int nlvls=4, int niter=100):
    """Samples for "niter" with "nlvls" or CRP and segments data.

    see seg.py's documentation
    """

    cdef IND_t i, j, iteration_number
    crps = []
    for i in range(nlvls):
        if i == 0:
            crps.append(CRP())
        else:
            crps.append(CRP(crps[i-1]))
    crps.reverse()
    crps = [None] + crps # concat [None] at head for easier indices alignment
    cdef int max_length = np.max([len(l) for l in data])
    cdef int nsentences = len(data)
    cdef int max_int = np.iinfo('i').max
    cdef INT32_t[:,:] segmentation = np.zeros((nsentences, max_length),
            dtype='int32')
    for i in range(nsentences):
        segmentation[i][len(data[i])-1] = max_int
    cdef int b_lvl, lvl, lower_lvl
    cdef INT32_t[:] line
    cdef float randnum, with_b, without_b
    ### This part is sample
    for iteration_number in range(niter):
        print "starting iteration:", iteration_number
        start_t = time.time()
        for i in range(nsentences):
            line = segmentation[i]  # that's ok, this is optimized by GCC
            for lvl in range(1, nlvls+1):
                lower_lvl = lvl - 1
                for j in range(max_length):
                    b_lvl = line[j]
                    if b_lvl == max_int:
                        break
                    if b_lvl < lower_lvl:
                        continue
                    if b_lvl >= lvl:
                        # remove the boundary by setting it at lower_lvl
                        line[j] = lower_lvl
                        for k in xrange(lvl, nlvls+1):
                            # find the segments/customers that are affected
                            ba = search_before_after(line, j, k)
                            # remove these customers
                            crps[k].removeCustomer(''.join(data[i][ba.before:j+1]))
                            crps[k].removeCustomer(''.join(data[i][j+1:ba.after]))
                    ba = search_before_after(line, j, lvl)
                    oneword = ''.join(data[i][ba.before:ba.after])
                    twowords = (''.join(data[i][ba.before:j+1]),
                            ''.join(data[i][j+1:ba.after]))
                    without_b = crps[lvl].predProb(oneword)  # without boundary
                    with_b = 0.5 * (crps[lvl].predProb(twowords[0]) +
                            crps[lvl].predProb(twowords[1])) # TODO correct
                    ###print oneword, without_b
                    ###print twowords, with_b
                    randnum = rand() / float(RAND_MAX)
                    if randnum > with_b/(without_b + with_b):
                        line[j] = lvl
                        crps[lvl].addCustomer(twowords[0])
                        crps[lvl].addCustomer(twowords[1])
                    else:
                        crps[lvl].addCustomer(oneword)
        print "took:", time.time() - start_t, "seconds"
    for lvl, crp in enumerate(crps):
        if crp != None:
            print "table at lvl:", lvl
            print crp._tables

    ### This part is segment
    # using just the last sample
    # TODO: minimum Bayes risk decoding (MAP over the CRPs distributions)
    ret = []
    cdef int prev_b
    for lvl in range(1, nlvls+1):
        corpus = []
        for i, line in enumerate(segmentation):
            prev_b = 0
            sentence = []
            ###print data[i]
            ###print [k for k in line]
            for j, b_lvl in enumerate(line):
                if b_lvl == max_int:
                    break
                if b_lvl >= lvl:
                    sentence.append(''.join(data[i][prev_b:j+1]))
                    prev_b = j+1
            sentence.append(''.join(data[i][prev_b:]))
            corpus.append(sentence)
        ret.append(corpus)
    return ret


