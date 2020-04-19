#!/bin/sh
HOSTFILE=$1
# for repeat in 1 2 3 4 5
# do
#     for lr in 0.1
#     do
#         for batchsize in 16
#         do
#             horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd.py \
#             --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
#             --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize}

#             sleep 60
#             bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
#             sleep 60
#         done
#     done
# done

for repeat in 1 2 3 4 5
do
    for lr in 0.1 0.2
    do
        for batchsize in 32
        do
            horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd.py \
            --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
            --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize}

            sleep 60
            bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
            sleep 60
        done
    done
done
