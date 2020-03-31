for lr in 0.01 0.05 0.1
do
    for batchsize in 32
    do
        horovodrun -np 4 -H localhost:4 python3 train_cifar100_hvd.py \
        --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
        --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize}
        sleep 60
    done
done
for lr in 0.01 0.05 0.1
do
    for batchsize in 32
    do
        horovodrun -np 4 -H localhost:4 python3 train_cifar100_hvd.py \
        --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
        --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize}
        sleep 60
    done
done