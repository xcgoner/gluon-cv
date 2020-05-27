#!/bin/sh
HOSTFILE=$1

for repeat in 1
do
    ############ sgd

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.1 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    --trainer sgd 

    sleep 60
    bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    sleep 60

    # ############ efsgd

    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer efsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 32

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60


    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer efsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 256

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60

    
    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer efsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 1024

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60

    # ############ qsparse

    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer qsparselocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 4 --local-sgd-interval 8

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60

    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer qsparselocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 32 --local-sgd-interval 8

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60

    # horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    # --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    # --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    # --model resnet50_v2 --mode hybrid   --lr 0.025 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    # --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    # --trainer qsparselocalsgd --input-sparse-1 1 --output-sparse-1 1 --layer-sparse-1 128 --local-sgd-interval 8

    # sleep 60
    # bash /home/ubuntu/src/ersgd/ec2-tools/pkill_cluster.sh ${HOSTFILE}-0
    # sleep 60
    
done
