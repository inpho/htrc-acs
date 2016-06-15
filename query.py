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

print "Retrieving HT IDs"
HOST = "thatchpalm.pti.indiana.edu"
PORT = 6079
BASE_URL = '/getid?category={0}'#'&number={1}'
COUNT_URL = '/findidcount?category={0}'

CONFIG_PATH = '/var/htrc-loc/config/'
if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)

for item in loader.keys()[:12]:
    time.sleep(1)
    cls = item.rsplit('/', 1)[-1]
    try:
        conn = httplib.HTTPConnection(HOST, PORT)
        conn.request('GET', BASE_URL.format(cls))
        r1 = conn.getresponse()
        if r1.status == 200:
            data = r1.read()
            htids = [i for i in data.split('|') if i]
            if htids:
                print "FOUND", cls, len(htids)
                output_path = os.path.join('/var/htrc-loc/samples/', cls)
                if len(htids) > 50 and len(htids) < 2500:
                    hathitrust.download_vols(htids, output_path)
    
                    os.chdir('/var/htrc-loc/config/')
                    seed = random.getrandbits(31)
                    config_file = "{cls}.ini".format(cls=cls,seed=seed)
   
                    # create corpus
                    subprocess.call(["vsm", "init", "--name", cls, 
                                     "--htrc", "--rebuild",
                                     output_path,
                                     config_file])

                    # filter language stoplists
                    hathitrust.get_metadata(output_path)
                    subprocess.call(["vsm", "prep", config_file, 
                                      "-q"])

                    # filter low and high 10 percent after language stopping
                    config = ConfigParser()
                    config.read(config_file)

                    c = Corpus.load(config.get("main", "corpus_file"))
                    low = get_low_filter(c, 0.10)
                    high = get_high_filter(c, 0.10)
                    print "FILTERING", low, high, "from", cls

                    subprocess.call(["vsm", "prep", config_file, 
                                      "--low", str(low),
                                      "--high", str(high), '-q']) 

                    for i in range(10):
                        seed = random.getrandbits(31)
                        seed_file = "{cls}.{seed}.ini".format(cls=cls,seed=seed)
                        shutil.copy(config_file, seed_file)
                        print "Copying", config_file, "-->", seed_file
    
                        # train model
                        subprocess.call(["vsm", "train", seed_file,
                                          "--context-type", "book",
                                          "--iter", "500",
                                          "--seed", str(int(seed)),
                                          "-k", "20", "40", "60", "80"])
            else:
                print "NONE ", cls
        else:
            raise httplib.HTTPException
    except httplib.HTTPException:
        print "ERROR", cls
    finally:
        conn.close()

