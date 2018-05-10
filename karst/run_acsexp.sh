HOME=/N/u/jammurdo/Karst
PATH=$HOME/anaconda3/bin/:$PATH

ACS_PATH=/N/dc2/projects/htrc-acs-loc
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
which topicexplorer

cd $HOME/acs/htrc-acs/
python acsexp.py --log-dir $ACS_PATH/results --samples 10 --iter 500 -k $K $INPHO_AREA
