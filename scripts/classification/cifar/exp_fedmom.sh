#!/bin/sh

for repeat in 1; do
    for H in 10; do
        for lr in 0.08 0.06 0.04 0.02; do
            horovodrun -np 8 -H localhost:8 python3 train_cifar_hvd_fedmom.py \
            --model cifar_wideresnet40_8 --local-optimizer nag --global-optimizer nag \
            --local-lr 0.1 --global-lr ${lr} --lr-decay 0.2 --lr-decay-epoch 60,120,160 \
            --wd 0.0005 --num-epochs 200 --batch-size 16 --local-sgd-interval ${H} \
            --mode hybrid -j 4 --nclasses 100

            sleep 10
            pkill -9 python3
            sleep 10

            horovodrun -np 8 -H localhost:8 python3 train_cifar_hvd_fedmom.py \
            --model cifar_wideresnet40_8 --local-optimizer sgd --global-optimizer nag \
            --local-lr 0.1 --global-lr ${lr} --lr-decay 0.2 --lr-decay-epoch 60,120,160 \
            --wd 0.0005 --num-epochs 200 --batch-size 16 --local-sgd-interval ${H} \
            --mode hybrid -j 4 --nclasses 100

            sleep 10
            pkill -9 python3
            sleep 10
        done
    done
done
