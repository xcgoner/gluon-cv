#!/bin/sh
HOSTFILE=$1
nworkers=$2
for repeat in 1
do
    for lr in 0.05
    do
        for batchsize in 16 32 64
        do
            HOROVOD_HIERARCHICAL_ALLREDUCE=1 horovodrun -np ${nworkers} -hostfile ${HOSTFILE} python3 train_cifar100_hvd.py \
            --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
            --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 15 --batch-size ${batchsize}  --test-throughput

            sleep 60
            bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
            sleep 60
        done
    done
done
