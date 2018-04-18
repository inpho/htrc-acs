from collections import namedtuple
from ConfigParser import RawConfigParser as ConfigParser
import httplib
import os
import os.path
import random
import shutil
import subprocess
import time

from topicexplorer.lib import hathitrust
import numpy as np
import rdflib
import skos
from vsm.corpus import Corpus

def get_items_counts(x):
    from scipy.stats import itemfreq
    try:
        # for speed increase with numpy >= 1.9.0
        items, counts = np.unique(x, return_counts=True)
    except:
        # for compatability
        ifreq = itemfreq(x)
        items = ifreq[:,0]
        counts = ifreq[:,1]
    return items, counts

def get_mask(c, words=None, filter=None):
    if filter is None:
        mask = np.ones(len(c.words), dtype=bool) # all elements included/True.
    else:
        mask = np.zeros(len(c.words), dtype=bool) # all elements excluded/False.
        mask[filter] = True

    if words:
        ix = np.in1d(c.words, list(words))
        ix = np.where(ix)
        mask[ix] = False              # Set unwanted elements to False

    return mask[:]

def get_high_filter(c, percent):
    items, counts = get_items_counts(c.corpus)
    bins = 1. - np.array([0., percent, 1.0])

    thresh = np.cumsum(counts[counts.argsort()]) / float(counts.sum())
    bins = [counts[counts.argsort()][np.searchsorted(thresh, bin)] for bin in bins]
    bins = sorted(set(bins))
    bins.append(max(counts))
    return bins[1]

def get_low_filter(c, percent):
    items, counts = get_items_counts(c.corpus)
    bins = 1. - np.array([0., percent, 1.0])

    thresh = np.cumsum(counts[counts.argsort()[::-1]]) / float(counts.sum())
    bins = [counts[counts.argsort()[::-1]][np.searchsorted(thresh, bin)] for bin in bins]
    bins = sorted(set(bins))

    return bins[1]


LCCO_URI = 'http://inkdroid.org/lcco/'
LCC = namedtuple('LCC', ['cls', 'subcls', 'topic'])

# create RDF graph
print "Loading RDF Graph"
graph = rdflib.Graph()
with open('data/lcco.rdf') as skosfile:
    graph.parse('data/lcco.rdf')

# create SKOS representation
print "Creating SKOS representation"
loader = skos.RDFLoader(graph)

areas = os.listdir('/var/htrc-loc/samples/')

for area in areas:
    a = loader.get('http://inkdroid.org/lcco/{}'.format(area), None)
    if a:
        print area, str(a.prefLabel)

