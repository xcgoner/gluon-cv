for rowsparse in 20
do
    for layersparse in 1 2 5
    do
        for lr in 0.01 0.05 0.1
        do
            for batchsize in 32
            do
                horovodrun -np 4 -H localhost:4 python3 train_cifar100_hvd_ersgd_v2.py \
                --model cifar_wideresnet40_8 --optimizer ersgdv2 --lr ${lr} --lr-decay 0.2 --nesterov \
                --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
                --row-sparse ${rowsparse} --layer-sparse ${layersparse} 
                sleep 60
            done
        done
    done
done
for rowsparse in 20
do
    for layersparse in 1 2 5
    do
        for lr in 0.01 0.05 0.1
        do
            for batchsize in 32
            do
                horovodrun -np 4 -H localhost:4 python3 train_cifar100_hvd_ersgd_v2.py \
                --model cifar_wideresnet40_8 --optimizer ersgdv2 --lr ${lr} --lr-decay 0.2 --nesterov \
                --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
                --row-sparse ${rowsparse} --layer-sparse ${layersparse} 
                sleep 60
            done
        done
    done
done