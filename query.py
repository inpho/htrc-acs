from collections import namedtuple
import httplib
import os.path
import time

from topicexplorer.lib import hathitrust
import rdflib
import skos

LCCO_URI = 'http://inkdroid.org/lcco/'
LCC = namedtuple('LCC', ['cls', 'subcls', 'topic'])

# create RDF graph
graph = rdflib.Graph()
with open('data/lcco.rdf') as skosfile:
    graph.parse('data/lcco.rdf')

# create SKOS representation
loader = skos.RDFLoader(graph)

HOST = "thatchpalm.pti.indiana.edu"
PORT = 6079
BASE_URL = '/getid?category={0}'#'&number={1}'
COUNT_URL = '/findidcount?category={0}'

for item in loader.keys()[:10]:
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
                output_path = os.path.join('/var/htrc-loc/', cls)
                if len(htids) > 10:
                    htids = htids[:10]
                hathitrust.download_vols(htids, output_path)
            else:
                print "NONE ", cls
        else:
            raise httplib.HTTPException
    except httplib.HTTPException:
        print "ERROR", cls
    finally:
        conn.close()

