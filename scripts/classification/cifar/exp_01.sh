for lr in 0.05 0.1 0.5
do
    for batchsize in 8 16 32
    do
        horovodrun -np 8 -H localhost:8 python3 train_cifar100_hvd.py \
        --model cifar_wideresnet40_8 --optimizer sgd --lr ${lr} --lr-decay 0.1 \
        --lr-decay-epoch 60,120 --wd 0.0005 --num-epochs 150 --batch-size ${batchsize}
        sleep 60

        horovodrun -np 8 -H localhost:8 python3 train_cifar100_hvd.py \
        --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.1 \
        --lr-decay-epoch 60,120 --wd 0.0005 --num-epochs 150 --batch-size ${batchsize}
        sleep 60
    done
done