#!/bin/sh
HOSTFILE=$1

for repeat in 1; do
    inputsparse1=32
    outputsparse1=8
    layersparse1=1
    inputsparse2=1
    for warmup in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; do
        for lr in 0.05 0.1; do
            for batchsize in 16; do
                let localsgdinterval=inputsparse1*outputsparse1*layersparse1/inputsparse2
                horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_ersgd2_v2.py \
                --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
                --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
                --input-sparse-1 ${inputsparse1} --output-sparse-1 ${outputsparse1} --layer-sparse-1 ${layersparse1} \
                --input-sparse-2 ${inputsparse2} --output-sparse-2 1 --layer-sparse-2 1 \
                --local-sgd-interval ${localsgdinterval} --warmup ${warmup}

                sleep 60
                bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
                sleep 60
            done
        done
    done
done
