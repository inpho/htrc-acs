
# coding: utf-8

# In[2]:


# In[4]:

from vsm import *
import numpy as np
import itertools
import copy
from scipy.stats import spearmanr, rankdata
from scipy.stats import pearsonr

from random import randrange
import random


import os.path
from ConfigParser import ConfigParser, NoOptionError

def is_valid_filepath(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


# In[3]:

# In[41]:

def deep_subcorpus(labels):
    bookssss = v.corpus.view_metadata('book')['book_label']
    print "{} total books".format(len(bookssss))
    if not all(d in bookssss for d in labels):
        raise ValueError("There is a book missing!")
    # resolve labels to indexes
    docs_labels = [v._res_doc_type(d) for d in labels]
    docs, labels = zip(*docs_labels)
    
    # get lengths of all contexts
    lens = np.array([len(ctx) for ctx in v.corpus.view_contexts('book')])
    
    # get the context_type index for use with context_data
    ctx_idx = v.corpus.context_types.index(v.model.context_type)
    
    # get original slices
    slice_idxs = [range(s.start,s.stop) for i, s in enumerate(v.corpus.view_contexts('book',as_slices=True)) 
                      if i in docs]
    
    new_corpus = copy.deepcopy(v.corpus)
    # reduce corpus to subcorpus    
    new_corpus.corpus = new_corpus.corpus[list(itertools.chain(*slice_idxs))]
    
    # reinitialize index fields
    for i,d in enumerate(docs):
        new_corpus.context_data[ctx_idx]['idx'][d] = lens[list(docs[:i+1])].sum()
    
    # reduce metadata to only the new subcorpus
    new_corpus.context_data[ctx_idx] = new_corpus.context_data[ctx_idx][list(docs)]
    
    return new_corpus



# In[44]:

"""
`vsm.extensions.comparison.lda`

Contains functions for comparing two topic models, represented as `LdaViewer` objects.
Also contains a visualization method `plot_topic_similarity`.
"""

from vsm.spatial import *

import numpy as np
import scipy.cluster.hierarchy as sch
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ['model_dist','avg_log_likelihood','perplexity','model_stats','plot_topic_similarity','compare_models']

def topic_overlap(v1,v2):
    """
    Calculates the overlap of two corpora and recalculates normalized topic matricies
    including only the overlapping words, ordered by the overlap order.
    
    Returns the joint vocabulary, v1.topics() and v2.topics() filtered and renormed.
    """
    vocab = set(v1.corpus.words)
    t1 = np.array(v1.topics())['value']
    t2 = np.array(v2.topics())['value']
    
    if v1.corpus.words_int != v2.corpus.words_int:
        print "corpus.words_int different, aligning words"
        vocab = vocab.intersection(v2.corpus.words)
        print "preserving {}% of words in v1; ".format(100 * len(vocab) / float(len(v1.corpus.words))),
        print "{}% of words in v2 ".format(100 * len(vocab) / float(len(v2.corpus.words)))
        
        t1 = t1[:,np.array([v1.corpus.words_int[word] for word in vocab])]
        t1 = (t1.T / t1.sum(axis=1)).T

        t2 = t2[:,np.array([v2.corpus.words_int[word] for word in vocab])]
        t2 = (t2.T / t2.sum(axis=1)).T

    return (vocab, t1, t2)

def model_dist(v1,v2, dist_fn=JS_dist):
    """
    Takes two LdaViewer objects and a distance metric and calculates the topic-topic distance.
    """
    vocab, t1, t2 = topic_overlap(v1,v2)
    #t1 = np.array(v1.topics())['value']
    #t2 = np.array(v2.topics())['value']
    combined = np.concatenate((t1,t2))
    
    # NOTE: Doing this by row to reduce memory requirements and time requirements
    D = np.column_stack(np.lib.pad(dist_fn(combined[i:,:],row.T), 
                                   (i,0), 'constant', constant_values=0)
                                if i + 1 < len(combined) else np.zeros(len(combined))
                            for i,row in enumerate(combined))
    return D + D.T - np.diag(D.diagonal())
    # Old simple version:
    # return np.column_stack(dist_fn(combined,row.T) for row in combined)

def doc_overlap(v1,v2, context_type, norm=True):
    context_label = context_type + '_label'
    ids = np.intersect1d(v1.corpus.view_metadata(context_type)[context_label], 
                         v2.corpus.view_metadata(context_type)[context_label])
    d1 = v1.doc_topic_matrix(ids)
    d2 = v2.doc_topic_matrix(ids)

    # renormalize so that each topic is now a document probability
    if norm:
        d1 = (d1 / d1.sum(axis=0))
        d2 = (d2 / d2.sum(axis=0))
    
    # original d1 and d2 are doc_topic, switch to topic_doc
    return (ids, d1.T, d2.T)

def model_doc_dist(v1, v2, context_type, dist_fn=JS_dist):
    ids, d1, d2 = doc_overlap(v1,v2, context_type)
    combined = np.concatenate((d1,d2))

    # NOTE: Doing this by row to reduce memory requirements and time requirements
    D = np.column_stack(np.lib.pad(dist_fn(combined[i:,:],row.T),
                                   (i,0), 'constant', constant_values=0)
                                    if i + 1 < len(combined) else np.zeros(len(combined))
                                for i,row in enumerate(combined))
    return D + D.T - np.diag(D.diagonal())


def pearson(v1, v2, context_type):
    context_label = context_type + '_label'
    ids, d1, d2 = doc_overlap(v1, v2, context_type)

    r_all = []
    for id in ids:
        sim1 = v1.dist_doc_doc(id)
        ix = np.in1d(sim1['doc'], ids).reshape(sim1['doc'].shape)
        sim1 = sim1[np.where(ix)]
        sim1 = sim1[sim1['doc'].argsort()]
        
        sim2 = v2.dist_doc_doc(id)
        ix = np.in1d(sim2['doc'], ids).reshape(sim2['doc'].shape)
        sim2 = sim2[np.where(ix)]
        sim2 = sim2[sim2['doc'].argsort()]
        
        r, pval = pearsonr(sim1['value'], sim2['value'])
        r_all.append(r)
    
    return sum(r_all)/len(r_all)

def spearman(v1, v2, context_type):
    context_label = context_type + '_label'
    ids, d1, d2 = doc_overlap(v1, v2, context_type)
    print len(ids), len(d1), len(d2)

    r_all = []
    for id in ids:
        sim1 = v1.dist_doc_doc(id)
        ix = np.in1d(sim1['doc'], ids).reshape(sim1['doc'].shape)
        sim1 = sim1[np.where(ix)]
        sim1 = sim1[sim1['doc'].argsort()]
        
        sim2 = v2.dist_doc_doc(id)
        ix = np.in1d(sim2['doc'], ids).reshape(sim2['doc'].shape)
        sim2 = sim2[np.where(ix)]
        sim2 = sim2[sim2['doc'].argsort()]

        r, pval = spearmanr(rankdata(sim1['value']), rankdata(sim2['value']))
        r_all.append(r)

    return sum(r_all)/len(r_all)

def recall(v1,v2, context_type,N=10):
    context_label = context_type + '_label'
    ids, d1, d2 = doc_overlap(v1, v2, context_type)

    r_all = []
    for id in ids:
        sim1 = v1.dist_doc_doc(id)
        ix = np.in1d(sim1['doc'], ids).reshape(sim1['doc'].shape)
        sim1 = sim1[np.where(ix)]

        sim2 = v2.dist_doc_doc(id)
        ix = np.in1d(sim2['doc'], ids).reshape(sim2['doc'].shape)
        sim2 = sim2[np.where(ix)]

        sim1 = np.array(sim1[1:N+2])['doc']
        sim2 = np.array(sim2[1:N+2])['doc']
        r = np.where(np.in1d(sim1, sim2))[0].size / float(N)
        r_all.append(r)
    
    return sum(r_all)/len(r_all)

def avg_log_likelihood(viewer):
    """ Calculates the average log likelihood per token. """
    return viewer.model.log_probs[-1][1] / len(viewer.corpus.corpus)

def perplexity(viewer):
    """ Calculates the perplexity. """
    return np.exp(-1*avg_log_likelihood(viewer))

def model_stats(*viewers):
    """
    Prints a table of avg log likelihood and perplexity for each viewer.
    """
    print "model", "k", "tokens", "types", "avg-log-likelihood", "perplexity"
    for i,v in enumerate(viewers):
        print "M{}".format(i), v.model.K, len(v.corpus.corpus), len(v.corpus.words), avg_log_likelihood(v), perplexity(v)

def create_dendogram(D, xdim=None, method='ward'):
    """
    Helper function for `plot_dendogram`. Sorts a matrix based on a hierarchical clustering method.
    .. see also: scipy.cluster.hierarchy
    """
    fig = figure(figsize=(3,9))
    Y = sch.linkage(D, method=method)
    Z = sch.dendrogram(Y, orientation='right')
    ax = fig.gca()
    ax.set_xticks([])
    #ax.set_yticks([])
    if xdim is not None:
        ax.set_yticklabels(["M2 " + str(idx - xdim) if idx >= xdim else "M1 " + str(idx)
                           for idx in np.array(Z['leaves'])])
    return Z

def plot_dendogram(D, Z, xdim,ydim, dist=None, filter_axis=False, show_self=False, alignment=None):
    # Compute and plot dendrogram.
    fig = figure(figsize=(12,10))
    
    D = np.copy(D)
    
    # Now make plot better
    if not show_self:
        D[:xdim,:xdim] = 0
        D[xdim:,xdim:] = 0

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = np.array(Z['leaves'])
    if filter_axis:
        D = D[index[index < xdim],:]
        D = D[:,index[index >= xdim]]
    else:
        D = D[index,:]
        D = D[:,index]
        
    # generate palette
    palette=cm.Blues_r
    palette.set_bad(alpha=0.0)
    MAX = np.sort(D.flatten())[int(.1*D.shape[0]*D.shape[1])]
    MAX = np.max(np.diagonal(dist[:,dist.argsort(axis=0)[1]]))
    MAX *= 1.1
    im = axmatrix.imshow(D, interpolation='none', cmap=palette, vmax=MAX, vmin=0.0)#aspect='auto', origin='lower')
    if filter_axis:
        axmatrix.set_xticks(arange(ydim))
        axmatrix.set_xticklabels([str(i - xdim) for i in index[index >= xdim]])
        axmatrix.set_yticks(arange(xdim))
        axmatrix.set_yticklabels([str(i) for i in index[index < xdim]])
    else:
        axmatrix.set_xticks(arange(xdim+ydim))
        axmatrix.set_xticklabels(index)
        axmatrix.set_yticks(arange(xdim+ydim))
        axmatrix.set_yticklabels(index)
    
    if alignment is not None:
        axmatrix.autoscale(False)
        ys,xs = zip(*alignment)
        xindex = index[index >= xdim] - xdim
        yindex = index[index < xdim]
        
        xs = np.array([np.squeeze(np.where(xindex == x)) for x in xs])
        ys = np.array([np.squeeze(np.where(yindex == y)) for y in ys])
        
        axmatrix.scatter(xs, ys, marker='o', s=125, color='w', lw=4, edgecolor='k')
    
    title("Jensen-Shannon Distance from topic to topic")
    ylabel("Model 1")
    xlabel("Model 2")
    # Plot colorbar.
    #axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    divider = make_axes_locatable(axmatrix)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    colorbar(im, cax=cax, extend='max')

def plot_topic_similarity(v1, v2, dist=None, dist_fn=JS_dist, sorted=True, alignment=None):
    # Calculate distance between all topics
    if dist is None:
        dist = model_dist(v1, v2, dist_fn)
    # print dist

    if sorted:
        dendo = create_dendogram(dist, xdim=v1.model.K, method='ward') 
        plot_dendogram(dist, dendo, v1.model.K, v2.model.K, dist=dist, filter_axis=True, alignment=alignment)
    else:
        # plot distance heatmap
        figure(figsize=(12,10))
        
        # filtering axis
        xdim = v1.model.K
        index = np.arange(0,len(dist))
        D = np.copy(dist)
        D = D[index[index < xdim],:]
        D = D[:,index[index >= xdim]]
        
        # generate palette
        palette=cm.Blues_r
        palette.set_bad(alpha=0.0)
        MAX = np.sort(D.flatten())[int(.1*D.shape[0]*D.shape[1])]
        MAX = np.max(np.diagonal(dist[:,dist.argsort(axis=0)[1]]))
        MAX *= 1.1
        
        # plot data
        im = imshow(D, cmap=palette, vmax=MAX, vmin=0.0, interpolation='none')
        if alignment is not None:
            ax = gca()
            ax.autoscale(False)
            ys,xs = zip(*alignment)
            scatter(xs,ys, marker='x', s=200, color='w', lw=5)
        
        # label data
        xticks(np.arange(D.shape[1]))
        yticks(np.arange(D.shape[0]))
        title("Jensen-Shannon Distance from topic to topic")
        xlabel("Model 1")
        ylabel("Model 2")
        
        # create heatmap
        divider = make_axes_locatable(gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        colorbar(im, cax=cax, extend='max')

def compare_models(v1, v2, context_type='document', dist_fn=JS_dist, sorted=True):
    model_stats(v1,v2)
    plot_topic_similarity(v1,v2,dist_fn=dist_fn,sorted=sorted)


# In[45]:

def alignment_fitness(topic_pairs, v1, v2, dist=None, dist_fn=JS_dist):
    """
    Takes a list of topic pair tuples and returns the sum of the JS_dist between them
    """    
    if dist is None:
        dist = model_dist(v1, v2, dist_fn)
    if dist.shape[0] == (v1.model.K + v2.model.K):
        dist = filter_dist(v1, v2, dist)
    
    return sum([dist[t[0]][t[1]] for t in topic_pairs])

def filter_dist(v1,v2,dist):
    xdim = v1.model.K
    index = np.arange(0,len(dist))
    D = np.copy(dist)
    D = D[index[index < xdim],:]
    D = D[:,index[index >= xdim]]
    return D

def plot_alignment(v1, v2, dist, alignment=None, fn_name=None):
    # Calculate distance between all topics
    if dist is None:
        dist = model_dist(v1, v2, dist_fn)
    if dist.shape[0] == (v1.model.K + v2.model.K):
        dist = filter_dist(v1, v2, dist)
    
    Xs = cm.jet_r(dist)
    
    if alignment is None:
        alpha = 1.0
    else:
        alpha = np.zeros(dist.shape)
        alpha[zip(*alignment)] = 1
    
    Xs[:,:,3] = alpha
    
    # plot distance heatmap
    figure(figsize=(12,10))
    imshow(Xs, interpolation='nearest', cmap='jet_r', vmin=0, vmax=1.0)
    colorbar()
    #imshow(alpha, interpolation=None, cmap=get_cmap('binary'), vmin=0, vmax=1.0, alpha=0.5)
    xticks(np.arange(dist.shape[1]))
    yticks(np.arange(dist.shape[0]))
    
    if fn_name is not None:
        title("Topic Alignment %s() using Jensen-Shannon Distance" % fn_name)
    else:
        title("Topic Alignment")
    xlabel("Model 1")
    ylabel("Model 2")
    show()


# In[46]:

def basic_alignment(v1, v2, dist=None, dist_fn=JS_dist, debug=False):
    """
    Simply aligns to the closest topic, allowing for multiple assignment. 
    Properties:
        non-surjective, non-injective
    """
    if dist is None:
        dist = model_dist(v1, v2, dist_fn)
    if dist.shape[0] == (v1.model.K + v2.model.K):
        dist = filter_dist(v1, v2, dist)
        
    alignment = []
    for i, topic in enumerate(dist):
        # topic = a[i]
        #s = topic[topic.argsort()]
        #sim = topic.argsort()[s < 0.05]
        closest = topic.argsort()[0]
        alignment.append((i, closest))
        if debug:
            print i, closest, topic[closest]
    
    return alignment


# In[47]:

def naive_alignment(v1, v2, dist=None, dist_fn=JS_dist, debug=False):
    """
    First naive overlap detector just goes to next closest element if the first topic has already been assigned
    
    Properties: 
        k1 < k2: injective, non-surjective
        k1 == k2: bijective
    """
    if v1.model.K > v2.model.K:
        raise ValueError("Models must have k1 <= k2")
    if dist is None:
        dist = model_dist(v1, v2, dist_fn)
    if dist.shape[0] == (v1.model.K + v2.model.K):
        dist = filter_dist(v1, v2, dist)
    
    alignment = []
    aligned = []
    for i, topic in enumerate(dist):
        # topic = a[i]
        #s = topic[topic.argsort()]
        #sim = topic.argsort()[s < 0.05]
        topic_idx = 0
        closest = topic.argsort()[topic_idx]
        if debug:
            print i, closest, topic[closest]
        
        while closest in aligned:
            topic_idx += 1
            closest = topic.argsort()[topic_idx]
            if debug:
                print i, closest, topic[closest]
        
        
        aligned.append(closest)
        alignment.append((i, closest))
    
    return alignment

def compare(sample_v, v, filename=None):
    sample_size = len(sample_v.labels)

    try:
        seed = sample_v.model.seed
    except AttributeError:
        seed = sample_v.model.seeds[0]

    try:
        span_seed = v.model.seed
    except AttributeError:
        span_seed = v.model.seeds[0]

    log_line = ''
    header_line = ''

    header_line = ['k', 'N', 'seed', 'span_seed', 'LL', 'corpus_size']
    log_line += "{k}\t{N}\t{seed}\t{span_seed}\t{LL}\t{corpus_size}\t".format(k=sample_v.model.K, 
        N=sample_size, seed=seed, 
        span_seed=span_seed,
        LL=sample_v.model.log_probs[-1][1],
        corpus_size=len(sample_v.corpus))

    # compute similarity on topic-word matrix - given a topic, what is its
    # distribution over words?
    dist = model_dist(sample_v, v)
    basic = basic_alignment(sample_v, v, dist=dist)
    naive = naive_alignment(sample_v, v, dist=dist)
    m1, m2 = zip(*basic)
    
    header_line.extend(['phi_fitness', 'phi_naive_fitness', 'phi_overlap'])
    log_line += "{fitness}\t{naive_fitness}\t{overlap}\t".format(
    	fitness=alignment_fitness(basic, sample_v, v, dist=dist),
    	naive_fitness=alignment_fitness(naive, sample_v, v, dist=dist),
    	overlap=len(set(m2)))

    # Compute similarity on topic-document matrix - given a topic, what is its
    # distribution over documents?
    dist = model_doc_dist(sample_v, v, 'book')
    basic = basic_alignment(sample_v, v, dist=dist)
    naive = naive_alignment(sample_v, v, dist=dist)
    m1, m2 = zip(*basic)
    
    header_line.extend(['theta_fitness', 'theta_naive_fitness', 'theta_overlap'])
    log_line += "{fitness}\t{naive_fitness}\t{overlap}\t".format(
    	fitness=alignment_fitness(basic, sample_v, v, dist=dist),
    	naive_fitness=alignment_fitness(naive, sample_v, v, dist=dist),
    	overlap=len(set(m2)))

    # Calculate Spearman, Pearson, top-10 recall, and top-10-percent recall
    # for each document - more of an IR-related search
    """
    print "{spearman}\t{pearson}\t{recall}\t{recall10p}".format(
        spearman=spearman(sample_v, v,'book'),
        pearson=pearson(sample_v, v, 'book'),
        recall=recall(sample_v, v, 'book', N=10),
        recall10p=recall(sample_v,v,'book', N=int(np.floor(0.1*sample_size)))),
    """

    header_line = '\t'.join(header_line)

    if filename is None:
        print log_line
    if filename is not None:
        write_header = not os.path.exists(filename)
            
        with open(filename, 'a') as logfile:
            if write_header:
                logfile.write(header_line + '\n')
            logfile.write(log_line + '\n')

def populate_parser(parser):
    parser.add_argument('config', type=lambda x: is_valid_filepath(parser, x),
        help="Configuration file path")
    parser.add_argument('-k', type=int, required=True,
        help="Number of Topics")
    parser.add_argument('--abort', action='store_true', dest='abort') 
    parser.add_argument('-f', '--force', action='store_false', dest='abort') 
    parser.set_defaults(abort=True)
    parser.add_argument('--samples', type=int, default=100,
        help="Number of Sample Models")
    parser.add_argument('--iter', type=int, default=200,
        help="Number of Iteratioins per training")

# In[48]:
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    populate_parser(parser)
    args = parser.parse_args()
    
    config = ConfigParser()
    config.read(args.config)

    # path variables
    path = config.get('main', 'path')
    corpus_file = config.get('main', 'corpus_file')

    c = Corpus.load(corpus_file)
    spanning_viewers = []
    ids = []

    from glob import iglob as glob
    area = os.path.basename(args.config)[:-4]
    glob_path = args.config.replace('.ini', '/*.ini')
    for config_path in glob(glob_path):
        print "loading", config_path
        config = ConfigParser()
        config.read(config_path)
        try: 
            model_pattern = config.get('main', 'model_pattern')
            context_type = config.get('main', 'context_type')
            if corpus_file != config.get('main', 'corpus_file'):
                raise ValueError('corpus_file not equal for' + config_path)
            m = LdaCgsMulti.load(model_pattern.format(args.k))
            v = LdaCgsViewer(c, m)
            spanning_viewers.append(v)
    
            if not len(ids):
                ids = v.corpus.view_metadata('book')['book_label']
                print "{} total books".format(len(ids))

        except NoOptionError:
            if args.abort:
                import sys
                print "area not yet done training spanning models"
                sys.exit()
            else:
                # if partial completion is allowed just move on to the
                # next area.
                pass

    
    if not os.path.exists('/var/htrc-loc/logs'):
        os.makedirs('/var/htrc-loc/logs')
    log_filename = "/var/htrc-loc/logs/{area}.results.log".format(k=args.k, area=area)
    for i in range(args.samples):
        # context_type = config.get('main', 'context_type')
        sample_size = randrange(int(len(ids)*0.1),int(1.0*len(ids)))
        sample_ids = random.sample(ids, sample_size) # see sample_size parameter at top
        sample_c = deep_subcorpus(sample_ids)

        num_iter = args.iter
        sample_m = LdaCgsMulti(sample_c, 'book', args.k, n_proc=8) # set num_topics parameter at top
        sample_m.train(num_iter, verbose=0) # set num_iter at top
        sample_v = LdaCgsViewer(sample_c, sample_m)

        for v2 in spanning_viewers:
            compare(sample_v, v2, filename=log_filename)
