HOME=/N/u/jammurdo/Karst
PATH=$HOME/anaconda3/bin/:$PATH
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
which topicexplorer

AREA=${INPHO_AREA##*/}
echo "$AREA"
AREA=${AREA%.htids.txt.ini}
DIRNAME=${INPHO_AREA%/*}
SEED=`od -A n -t d -N 4 /dev/urandom | tr -d ' ' | tr -d '-'`
NEW_CONFIG=$DIRNAME/datasets/$AREA/$AREA.$SEED.ini

echo "PROCESSING: $AREA (seed: $SEED)"
mkdir -p $DIRNAME/datasets/$AREA
cp $INPHO_AREA $NEW_CONFIG

/usr/bin/time -v topicexplorer train -k 25 50 100 250 500 --iter 500 -p 16 --rebuild --seed $SEED -q $NEW_CONFIG

