#!/usr/bin/python
#-----------------------------------------------------------------------
# File    : trains.py
# Contents: spike train modification functions
# Authors : Sebastien Louis, Christian Borgelt
# History : 2009.08.?? file created
#           2009.08.20 spike train dithering improved
#           2009.08.24 spike shifting improved (general ranges)
#           2009.08.25 interspike interval dithering added
#           2009.09.02 changes for compatibility with Python 2.5.2
#-----------------------------------------------------------------------
from bisect  import bisect      # for binary search in sorted lists
from random  import random, randint, sample, choice
import numpy as np

#-----------------------------------------------------------------------
# Spike Train Functions
#-----------------------------------------------------------------------

def randomized (train, rng=range(1000)):
    '''Create a randomized (binned) spike train
    (that is, assign random new spike time bin indices).
    train: spike train to randomize
    rng:   range of allowed time bin indices'''
    return sample(rng, len(train))

# -----------------------------------------------------------------------


def from_spike_trains_to_spike_nums(spike_trains):
    min_time, max_time = get_range_train_list(spike_trains)
    n_times = int(np.ceil((max_time-min_time)+1))
    spike_nums = np.zeros((len(spike_trains), n_times), dtype="int16")
    for train_index, train in enumerate(spike_trains):
        # normalizing
        train = train - min_time
        # rounding to integers
        train = train.astype(int)
        # mask = np.zeros(n_times, dtype="bool")
        # mask[train] = True
        spike_nums[train_index, train] = 1
    return spike_nums

def from_spike_nums_to_spike_trains(spike_nums):
    spike_trains = []
    for spikes in spike_nums:
        spike_trains.append(np.where(spikes)[0])
    return spike_trains

def get_range_train_list(train_list):
    """

    :param train_list:
    :return: the min and max value of the train list
    """
    min_value = 0
    max_value = 0

    for i, train in enumerate(train_list):
        if len(train) == 0:
            return
        if i == 0:
            min_value = np.min(train)
            max_value = np.max(train)
            continue
        min_value = min(min_value, np.min(train))
        max_value = max(max_value, np.max(train))

    return min_value, max_value


def shifted (train, shift=20, rng=range(1000)):
    '''Create a randomly shifted/rotated (binned) spike train.
    train: spike train to shift
    shift: maximum amount by which to shift the spikes
    rng:   range of allowed time bin indices
    returns a spike trains with the same number of spikes'''
    off = rng[0]
    n = rng[-1] + 1 - off
    shift = int(abs(shift)) % n  # compute the shift value
    shift = randint(-shift, shift) + n - off
    return [((x+shift) % n) + off for x in train]

#-----------------------------------------------------------------------

def dithered_fast (train, dither=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    dither: maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    d = [randint(max(x-dither,rng[ 0]),
                 min(x+dither,rng[-1])) for x in train]
    return [x for x in set(d)]  # remove possible duplicates

#-----------------------------------------------------------------------

def dithered (train, dither=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    dither: maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    d = [randint(max(x-dither,rng[ 0]),
                 min(x+dither,rng[-1])) for x in train]
    # This way of writing the dithering (randint call) ensures that
    # all new time bin indices lie in the allowed time bin index range.
    # The execution speed penalty for this is relatively small.
    if len(set(d)) == len(d): return d
    s = set()                   # check wether all bin indices differ
    for i in range(len(d)):    # if not, re-dither the duplicates
        if d[i] not in s: s.add(d[i]); continue
        r = range(max(rng[ 0],train[i]-dither),
                   min(rng[-1],train[i]+dither))
        r = [x for x in r if x not in s]
        if r: d[i] = choice(r); s.add(d[i])
    return d if len(d) == len(s) else [x for x in s]
    # Simply returning the initially created d may lose spikes,
    # because the initial d may contain duplicate bin indices.
    # This version tries to maintain the spike count if possible.
    # Only if all bins in the jitter window around some spike are
    # already taken, the spike cannot be dithered and is dropped.

#-----------------------------------------------------------------------

def isi_dithered (train, cdfs, isid=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    cdfs:   cumulative distribution functions for the
            interspike interval pairs with the same sum
    isid:   maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    train = [rng[0]-1] +train +[rng[-1]+1]
    size  = len(cdfs)           # expand train by dummy spikes and
    out   = set()               # initialize the output spike set
    for i in range(1,len(train)-1):
        a,s,b = train[i-1],train[i],train[i+1]
        x = s-a; k = x+(b-s)-2  # get three consecutive spikes
        if k < size:            # if inside range of distributions
            d = cdfs[k]         # dither according to isi distribution
            x = max(0,x-isid-1); y = min(len(d)-1,x+isid)
            for j in range(isid+isid):
                k = a +bisect(d[x:y],d[x]+random()*(d[y]-d[x]))
                if k not in out: out.add(k); break
        else:                   # if outside range of distributions
            r = range(max(a+1,s-isid),min(b-1,s+isid))
            r = [x for x in r if x not in out]
            if r: out.add(choice(r)) # dither with uniform distribution
    return [s for s in out]     # return the dithered spike train
