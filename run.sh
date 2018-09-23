#/bin/bash
# set -x
# ./run.sh DATASET train attack eval grid num_dx eps epoch_for_eval session_name

SESSION_NAME="neural_fingerprint"

DATADIR=/tmp/data/$1
BASE_LOGDIR=/tmp/logs/$SESSION_NAME/$1
mkdir -p $BASE_LOGDIR

# Define a grid of hyperparameters or use from flags
if [ "$5" = "grid" ]; then
ALL_NUMDX="5 10 30"
ALL_EPS="0.003 0.006 0.01"
EPOCHS="20"
else
ALL_NUMDX=$6
ALL_EPS=$7
EPOCHS=$8
fi

if [ "$1" = "mnist" ]; then
NUM_EPOCHS=1
EPOCHS="1"
fi


if [ "$1" = "cifar" ]; then
NUM_EPOCHS=50
fi

# Loop over grid of hyperparameters
for EPS in $ALL_EPS; do
for NUMDX in $ALL_NUMDX; do

LOGDIR=$BASE_LOGDIR/eps_$EPS/numdx_$NUMDX
mkdir -p $LOGDIR
mkdir -p $LOGDIR/ckpt
mkdir -p $LOGDIR/train
mkdir -p $LOGDIR/eval
mkdir -p $LOGDIR/adv_examples

# Train
if [ "$2" = "train" ]; then

python2 $1/train_fingerprint.py \
--batch-size 128 \
--test-batch-size 128 \
--epochs $NUM_EPOCHS \
--lr 0.001 \
--momentum 0.9 \
--seed 0 \
--log-interval 10 \
--log-dir $LOGDIR \
--data-dir $DATADIR \
--eps=$EPS \
--num-dx=$NUMDX \
--num-class=10 \
--name="$1"

fi


if [ "$1" = "cifar" ] && [ $5 = "nogrid" ]; then
epoch_file=$LOGDIR"/termination_epoch"
echo $epoch_file
while IFS= read -r var
do
  EPOCHS=$var
done < "$epoch_file"
echo $epochs":Number of epochs run"
fi


for EPOCH in $EPOCHS; do

ADV_EX_DIR=$LOGDIR/adv_examples/epoch_$epoch
mkdir -p $ADV_EX_DIR


# Generate attacks
if [ "$3" = "attack" ]; then

for ATCK in 'adapt-fgsm' 'cw-fp'; do #'spsa' 'adapt-fgsm'; do #  'all'; do
# Write out the different attacks, use 'all' to generate all attacks for same random subset of test
# For now copy paste this list into eval_fingerprint :) Will figure out a fix later

python2 $1/gen_whitebox_adv.py \
--attack $ATCK \
--ckpt $LOGDIR/ckpt/state_dict-ep_$EPOCH.pth \
--log-dir $ADV_EX_DIR \
--fingerprint-dir $LOGDIR \
--batch-size 128
done

fi


# Evaluate fingerprint perf on adversarial examples
# Evaluate fingerprint perf on adversarial examples
if [ "$4" = "eval" ]; then

EVAL_LOGDIR=$LOGDIR/eval/epoch_$EPOCH
mkdir -p $EVAL_LOGDIR

python2 $1/eval_fingerprint.py \
--batch-size 128 \
--epochs 100 \
--lr 0.001 \
--momentum 0.9 \
--seed 0 \
--log-interval 10 \
--ckpt $LOGDIR/ckpt/state_dict-ep_$EPOCH.pth \
--log-dir $EVAL_LOGDIR \
--fingerprint-dir $LOGDIR \
--adv-ex-dir $ADV_EX_DIR \
--data-dir $DATADIR \
--eps=$EPS \
--num-dx=$NUMDX \
--num-class=10 \
--name="$1" \
# --no-cuda
# --verbose
# --debug

fi


done


done
done
