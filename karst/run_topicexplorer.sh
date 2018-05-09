HOME=/N/u/jammurdo/Karst
PATH=$HOME/anaconda3/bin/:$PATH
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
which topicexplorer

echo "PROCESSING: $INPHO_AREA"
/usr/bin/time -v topicexplorer train -k 25 50 100 250 500 --iter 500 -p 16 --rebuild -q $INPHO_AREA

