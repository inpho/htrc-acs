# HTRC ACS Project
This repository contains code for the 2015 HathiTrust Reseach Center (HTRC) Advanced Collaborative Support (ACS) grant progam.

## Contents
`query.py` contains the code to download all columns in an LoC area and train spanning models.
`acsexp.py` contains the code to train sample models and compare them to the spanning models.

## Data Generated
- Corpus details:
  - **k** -- number of topics
  - **N** -- number of volumes in sample corpus
  - **sample_seed** -- seed for sample model
  - **spanning_seed** -- seed for spanning model
  - **LL** -- log likelihood of sample model
  - **num_tokens** -- number of tokens in sample corpus
- Topic-word matrix comparisons
  - **fitness** -- average distance between each topic pair to the closest topic, allowing duplicate assignments so not all topics are guaranteed a match.
  - **naive_fitness** -- average distance between each topic pair, without replacement, giving a 1-to-1 alignment.
  - **overlap** -- percent of topics captured by basic alignment
- Topic-document matrix comparisons
  - **fitness** -- average distance between each topic pair to the closest topic, allowing duplicate assignments so not all topics are guaranteed a match.
  - **naive_fitness** -- average distance between each topic pair, without replacement, giving a 1-to-1 alignment.
  - **overlap** -- percent of topics captured by basic alignment
- Overall similarity comparisons (Information retrieval)
  - **spearman** -- Spearman Rank Correlation of document similarity
  - **pearson** -- Pearson Correlation of document similarity
  - **recall** -- Percentage overlap (should always be 1.0)
  - **top10recall** -- Percentage overlap of similar docs in top 10 percent


## Datasets
htrc-loc-graph/marcrecs.py	http://www.loc.gov/cds/products/product.php?productID=5	lccn-oclc-lcco-alignment.tsv
htrc-loc-graph/htrc.py	https://www.hathitrust.org/hathifiles	htid-htrecord-oclc-lccn-alignment.tsv
htrc-loc-graphlcco_titles.py	data/lcco.rdf	lcco_titles.tsv
htrc-acs/import.sql	lccn-oclc-lcco-alignment.tsv;htid-htrecord-oclc-lccn-alignment.tsv	alignment.sqlite
htrc-acs/align.py

## Runtime
topicexplorer init $AREA --htrc --name "$(grep lcco_titles.py $AREA | awk '{print $1}')"
topicexplorer prep $AREA 
topicexplorer train $AREA -k 20 40 60 80 --iter 500 -p 24

htrc-acs/acsexp.py config_path -k $0 --samples 100 --iter 200

