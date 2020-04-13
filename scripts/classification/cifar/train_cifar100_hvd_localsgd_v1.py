import matplotlib
matplotlib.use('Agg')

import argparse, time, logging, random, os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

from gluoncv.data.sampler import SplitSampler

import horovod.mxnet as hvd

from gluoncv.trainer.local_sgd_trainer_v1 import LocalSGDTrainerV1

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--model', type=str, default='resnet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='ERSGDV1',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    parser.add_argument('--local-sgd-interval', type=int, default=4,
                        help='interval for model synchronization')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    hvd.init()
    
    batch_size = opt.batch_size
    classes = 100

    # num_gpus = opt.num_gpus
    # batch_size *= max(1, num_gpus)
    context = [mx.gpu(hvd.local_rank())]
    num_workers = hvd.size()
    rank = hvd.rank()

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)
    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx = context)
    # optimizer = 'nag'
    optimizer = opt.optimizer

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    # plot_path = opt.save_plot_dir

    logging.basicConfig(level=logging.INFO,
                    filename="train_cifar100_qsparselocalsgd_{}_{}_{}_{}.log".format(opt.model, opt.optimizer, opt.batch_size, opt.lr),
                    filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    if rank == 0:
        logging.info(opt)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            # outputs = [net(X) for X in data]
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        # if opt.print_tensor_shape and rank == 0:
        #     print(net)

        train_dataset = gluon.data.vision.CIFAR100(train=True).transform_first(transform_train)

        train_data = gluon.data.DataLoader(
            train_dataset,
            sampler=SplitSampler(len(train_dataset), num_parts=num_workers, part_index=rank),
            batch_size=batch_size, last_batch='discard', num_workers=opt.num_workers)

        # val_dataset = gluon.data.vision.CIFAR100(train=False).transform_first(transform_test)
        # val_data = gluon.data.DataLoader(
        #     val_dataset,
        #     sampler=SplitSampler(len(val_dataset), num_parts=num_workers, part_index=rank),
        #     batch_size=batch_size, num_workers=opt.num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)

        hvd.broadcast_parameters(net.collect_params(), root_rank=0)

        trainer = LocalSGDTrainerV1(
            net.collect_params(),  
            'nag', optimizer_params, 
            local_sgd_interval=opt.local_sgd_interval)

        # trainer = gluon.Trainer(net.collect_params(), optimizer,
                                # {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        lr = opt.lr

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                lr *= lr_decay
                trainer.set_learning_rate(lr)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1

            mx.nd.waitall()
            toc = time.time()
            
            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            # name, val_acc = test(ctx, val_data)

            trainer.pre_test()
            name, val_acc = test(ctx, val_data)
            trainer.post_test()
            
            train_history.update([1-acc, 1-val_acc])
            # train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

            # allreduce the results
            allreduce_array_nd = mx.nd.array([train_loss, acc, val_acc])
            hvd.allreduce_(allreduce_array_nd, name='allreduce_array', average=True)
            allreduce_array_np = allreduce_array_nd.asnumpy()
            train_loss = np.asscalar(allreduce_array_np[0])
            acc = np.asscalar(allreduce_array_np[1])
            val_acc = np.asscalar(allreduce_array_np[2])

            if val_acc > best_val_score:
                best_val_score = val_acc
                # net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

            if rank == 0:
                logging.info('[Epoch %d] train=%f val=%f loss=%f comm=%.2f time: %f' %
                    (epoch, acc, val_acc, train_loss, trainer._comm_counter/1e6, toc-tic))

                if save_period and save_dir and (epoch + 1) % save_period == 0:
                    net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

            trainer._comm_counter = 0.

        if rank == 0:
            if save_period and save_dir:
                net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))



    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
