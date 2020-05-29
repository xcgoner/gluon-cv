#!/bin/sh
HOSTFILE=$1

for repeat in 1
do

    ############ ersgd2

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd2 --input-sparse-1 1 --output-sparse-1 2 --layer-sparse-1 32    \
    --input-sparse-2 1 --output-sparse-2 1 --layer-sparse-2 4 --local-sgd-interval 16    \
    --sync-states

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd2 --input-sparse-1 1 --output-sparse-1 8 --layer-sparse-1 64    \
    --input-sparse-2 1 --output-sparse-2 4 --layer-sparse-2 4 --local-sgd-interval 32    \
    --sync-states

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd2 --input-sparse-1 1 --output-sparse-1 16 --layer-sparse-1 128    \
    --input-sparse-2 1 --output-sparse-2 4 --layer-sparse-2 2 --local-sgd-interval 256    \
    --sync-states

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60


    ############ ersgd


    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 32

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60


    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd --input-sparse-1 1 --output-sparse-1 4 --layer-sparse-1 64

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    
    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer ersgd --input-sparse-1 1 --output-sparse-1 8 --layer-sparse-1 128

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60


    ############ partial

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer partiallocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 8 --local-sgd-interval 4

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer partiallocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 16 --local-sgd-interval 16

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 0 --use-rec --dtype float16 --optimizer nag \
    --trainer partiallocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 32 --local-sgd-interval 32

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    
done
