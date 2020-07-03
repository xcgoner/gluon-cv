#!/bin/sh
nworkers=$1
for repeat in 1
do
    for lr in 0.05
    do
        for batchsize in 16 32 64
        do
            horovodrun -np ${nworkers} -H localhost:${nworkers} python3 train_cifar100_hvd.py \
            --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
            --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 15 --batch-size ${batchsize}

            sleep 60
            pkill -9 python3
            sleep 60
        done
    done
done
