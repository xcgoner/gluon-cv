#!/bin/sh

for repeat in 1; do
    for H in 10; do
        for globalmom in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05; do
            horovodrun -np 8 -H localhost:8 python3 train_cifar_hvd_doublemom.py \
            --model cifar_wideresnet40_8 --local-optimizer nag --global-optimizer nag \
            --local-lr 0.1 --global-lr 0.9 --lr-decay 0.2 --lr-decay-epoch 30,60,80 \
            --local-momentum 0.9 --global-momentum 0.9 \
            --wd 0.0005 --num-epochs 100 --batch-size 16 --local-sgd-interval ${H} \
            --mode hybrid -j 4 --nclasses 100

            sleep 10
            pkill -9 python3
            sleep 10

            # horovodrun -np 8 -H localhost:8 python3 train_cifar_hvd_doublemom.py \
            # --model cifar_wideresnet40_8 --local-optimizer sgd --global-optimizer nag \
            # --local-lr 0.1 --global-lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160 \
            # --local-momentum 0 --global-momentum 0.9 \
            # --wd 0.0005 --num-epochs 200 --batch-size 16 --local-sgd-interval ${H} \
            # --mode hybrid -j 4 --nclasses 100

            # sleep 10
            # pkill -9 python3
            # sleep 10
        done
    done
done
