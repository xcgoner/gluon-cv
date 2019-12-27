for i in 1 2 3 4 5
do
    for lr in 0.05 0.1
    do
        for batchsize in 8 16 32
        do
            for sparse in 0 0.2 0.4
            do 
                horovodrun -np 8 -H localhost:8 python3 train_cifar100_hvd_compr.py --model cifar_wideresnet40_8 \
                --optimizer efsgd --lr ${lr} --lr-decay 0.1 --lr-decay-epoch 60,120 --wd 0.0005 --reset-interval 0 \
                --sparse-ratio 0 --sparse-lower --compress --num-epochs 150 --batch-size ${batchsize}
                sleep 60
            done
        done
    done
done