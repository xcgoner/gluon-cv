#!/bin/sh
HOSTFILE=$1

for repeat in 1
do
    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 1 --output-sparse 4 --layer-sparse 1 --local-sgd-interval 8

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.05 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 32 --output-sparse 4 --layer-sparse 1 --local-sgd-interval 2

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.05 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 32 --output-sparse 4 --layer-sparse 1 --local-sgd-interval 8

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.05 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 32 --output-sparse 4 --layer-sparse 1 --local-sgd-interval 4

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 1 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 2

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 1 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 4

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 1 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 8

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.1 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 4 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 4

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.05 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 16 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 4

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
    --model cifar_wideresnet40_8 --optimizer nag --lr 0.05 --lr-decay 0.2 \
    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size 16 \
    --input-sparse 16 --output-sparse 1 --layer-sparse 1 --local-sgd-interval 8

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

done

# for repeat in 1
# do
#     for inputsparse in 1 4 16
#     do
#         for H in 2 4 8 
#         do
#             for lr in 0.05 0.1
#             do
#                 for batchsize in 16
#                 do
#                     horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
#                     --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
#                     --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
#                     --input-sparse ${inputsparse} --output-sparse 1 --layer-sparse 1 --local-sgd-interval ${H}

#                     sleep 60
#                     bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
#                     sleep 60
#                 done
#             done
#         done
#     done
# done

# for repeat in 1
# do
#     for inputsparse in 32
#     do
#         for H in 2 4 8
#         do
#             for lr in 0.05 0.1
#             do
#                 for batchsize in 16
#                 do
#                     horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_qsparselocalsgd_v1.py \
#                     --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
#                     --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
#                     --input-sparse ${inputsparse} --output-sparse 4 --layer-sparse 1 --local-sgd-interval ${H}

#                     sleep 60
#                     bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
#                     sleep 60
#                 done
#             done
#         done
#     done
# done