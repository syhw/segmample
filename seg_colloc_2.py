import numpy as np
import time 
from random import random
from collections import defaultdict
from chinese_restaurants import DummyCRP, CRP, pipe_it, word_at_lvl
DEBUG = True
BURNIN = 100

def sample_and_segment(data, nlvls=4, niter=100):
    """Samples for "niter" with "nlvls" or CRP and segments data.

    see seg.py's documentation
    """

    crps = []
    for i in range(nlvls):
        if i == 0:
            crps.append(CRP())
        else:
            crps.append(CRP(crps[i-1]))
    crps = [None] + crps # concat [None] at head for easier indices alignment
    max_length = np.max([len(l) for l in data])
    nsentences = len(data)
    max_int = np.iinfo('i').max
    segmentation = np.zeros((nsentences, max_length),
            dtype='int32')
    for i in range(nsentences):
        segmentation[i][len(data[i])-1] = max_int
    marginal_map_counts = [defaultdict(int) for _ in xrange(len(data))]

    ### This part is sample
    for iteration_number in range(niter):
        print "starting iteration:", iteration_number
        start_t = time.time()

        for i in xrange(nsentences):
            line = segmentation[i]
            if iteration_number > 0:
                for plvl in xrange(nlvls, 0, -1):
                    # from line and data[i] we want a list of strings (one for
                    # each colloc of lvl "plvl") with '|' for each colloc of lvl
                    # "plvl-1" if plvl > 0
                    previous_boundary = 0
                    los = []
                    for k, boundary in enumerate(line):
                        kk = k+1
                        if boundary == max_int:
                            break
                        if boundary >= plvl:
                            with_pipes = word_at_lvl(data[i], 
                                    previous_boundary, kk, line, plvl)
                            los.append(with_pipes)
                            previous_boundary = kk
                    with_pipes = pipe_it(data[i], previous_boundary, kk, line, plvl)
                    los.append(with_pipes)
                    for word_lvl in los:
                        crps[plvl].removeCustomer(word_lvl)

            total_length = 0
            for j, e in enumerate(line):
                if e == max_int:
                    total_length = j+1
                    break
                line[j] = 0
            for plvl in xrange(1, nlvls+1):
                print "PLVL =", plvl
                # build the chart for line & plvl
                length = total_length
                #if plvl > 1:
                #    length = np.sum(line[:total_length] == plvl-1)
                chart = np.zeros((length, length+1), dtype='float32')
                total = 0.0  # that will go to chart[column, length]
                for strlen in xrange(1, length+1):
                    if not(strlen == length) and not(line[strlen-1] == plvl-1):
                        #print "SKIP", strlen, length, plvl, line
                        continue
                    for lwl in xrange(1, strlen+1):
                        bound_pos = strlen-lwl
                        if not(bound_pos == 0) and not(line[bound_pos-1] == plvl-1):
                            print "SKIP", strlen, length, plvl, line
                            continue
                        if strlen == lwl:
                            wProb = crps[plvl].predProb(word_at_lvl(data[i], bound_pos, strlen, line, plvl))
                            print "AAAAAAAAAAAAAAAA", wProb
                        if strlen != lwl:
                            wProb = crps[plvl].predProb(word_at_lvl(data[i], bound_pos, strlen, line, plvl))
                            wProb = (wProb * chart[bound_pos-1, length] * 0.5)
                        chart[strlen-1, lwl-1] = wProb
                        total += wProb
                    chart[strlen-1, length] = total
                print chart

                # block sample
                strlen = length
                while strlen > 0:
                    probs = chart[strlen-1, :strlen]
                    flip = random() * chart[strlen-1, length]
                    cur = 0.0
                    final_ind = 0
                    for kk, entry in enumerate(probs):
                        cur += entry
                        if flip <= cur:
                            final_ind = kk
                            break
                    new_bound_pos = strlen - final_ind - 2
                    print "we sampled:", word_at_lvl(data[i], strlen-final_ind-1, strlen, line, plvl)
                    print "line before:", line
                    if new_bound_pos > -1:
                        line[new_bound_pos] = plvl  # TODO unhack
                    print "line after:", line
                    crps[plvl].addCustomer(word_at_lvl(data[i], strlen-final_ind-1, strlen, line, plvl))
                    strlen -= (final_ind + 1)
        if DEBUG:
            for lvl, crp in enumerate(crps):
                if crp != None:
                    print "table at lvl:", lvl
                    crp.verify()
        print "took time:", time.time() - start_t
        if iteration_number >= BURNIN:
            for i, utt in enumerate(segmentation):
                marginal_map_counts[i][tuple(utt)] += 1

    for lvl, crp in enumerate(crps):
        if crp != None:
            print "table at lvl:", lvl
            print crp._tables


    mms = [] 
    for i in xrange(len(data)):
        mms.append(max(marginal_map_counts[i].items(), key=lambda x:x[1])[0])
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


