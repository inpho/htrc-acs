#!/bin/bash
HOME=/N/u/jammurdo/Karst
PATH=$HOME/anaconda3/bin/:$PATH
ACS_PATH=/N/dc2/projects/htrc-acs-loc
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8

AREA=$1

qsub -N htrc-loc.$AREA.span -o $ACS_PATH/logs/$AREA.out -e $ACS_PATH/logs/$AREA.err -m ae -M jammurdo@indiana.edu -l walltime=12:00:00 -l nodes=1:ppn=16 -v INPHO_AREA=$ACS_PATH/$AREA.htids.txt.ini run_topicexplorer.sh

