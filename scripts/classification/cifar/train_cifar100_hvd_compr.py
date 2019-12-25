import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

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

from distributed_2stpes_trainer import Distributed2StepsTrainer

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
    parser.add_argument('--optimizer', type=str, default='signum', choices=['signum', 'efsgd', 'ersgd'],
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--nesterov', action='store_true',
                        help='use Nesterov momentum')
    parser.add_argument('--compress', action='store_true',
                        help='use 1-bit compression')
    parser.add_argument('--reset-interval', type=int, default=0,
                        help='period of error reset.')
    parser.add_argument('--sparse-ratio', type=float, default=0,
                        help='propotion of sparsity')
    parser.add_argument('--sparse-lower', action='store_true',
                        help='sparsify lower or higher layers')
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

    logging.basicConfig(level=logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("train_cifar100_{}_{}_{}_{}_{}_{}_{}_{}_{}.log".format(opt.model, opt.optimizer, opt.batch_size, opt.lr, opt.nesterov, opt.compress, opt.reset_interval, opt.sparse_ratio, opt.sparse_lower)),
            logging.StreamHandler()
        ])
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

    save_prev_lr = False
    if str.lower(optimizer) == 'signum':
        optimizer = 'signumpost'
        pre_optimizer = 'signumpre'
    elif str.lower(optimizer) == 'efsgd':
        optimizer = 'efsgdpost'
        pre_optimizer = 'efsgdpre'
        save_prev_lr = True
        optimizer_params['compress'] = opt.compress
    elif str.lower(optimizer) == 'ersgd':
        optimizer = 'ersgdpost'
        pre_optimizer = 'ersgdpre'
        save_prev_lr = False
        optimizer_params['nesterov'] = opt.nesterov
        optimizer_params['compress'] = opt.compress
        assert(opt.reset_interval > 0)
    elif str.lower(optimizer) == 'spsgd':
        optimizer = 'spsgdpost'
        pre_optimizer = 'spsgdpre'
        save_prev_lr = False
        optimizer_params['nesterov'] = opt.nesterov
        assert(opt.reset_interval > 0)
    else:
        pre_optimizer = None

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        train_dataset = gluon.data.vision.CIFAR100(train=True).transform_first(transform_train)

        val_dataset = gluon.data.vision.CIFAR100(train=False).transform_first(transform_test)

        train_data = gluon.data.DataLoader(
            train_dataset,
            sampler=SplitSampler(len(train_dataset), num_parts=num_workers, part_index=rank),
            batch_size=batch_size, last_batch='discard', num_workers=opt.num_workers)

        val_data = gluon.data.DataLoader(
            val_dataset,
            sampler=SplitSampler(len(val_dataset), num_parts=num_workers, part_index=rank),
            batch_size=batch_size, num_workers=opt.num_workers)
        
        # # allreduce val acc is not working
        # val_data = gluon.data.DataLoader(
        #     val_dataset,
        #     batch_size=batch_size, num_workers=opt.num_workers)

        hvd.broadcast_parameters(net.collect_params(), root_rank=0)

        trainer = Distributed2StepsTrainer(
            net.collect_params(),  
            optimizer,
            pre_optimizer, 
            optimizer_params, 
            save_prev_lr=save_prev_lr,
            reset_interval=opt.reset_interval,
            sparse_ratio=opt.sparse_ratio,
            sparse_lower=opt.sparse_lower)

        # trainer = gluon.Trainer(net.collect_params(), optimizer,
                                # {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
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

            if opt.reset_interval > 0:
                trainer.allreduce_params()
                trainer.allreduce_states()

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            
            train_history.update([1-acc, 1-val_acc])
            # train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

            # why average=False ???
            train_loss_nd = mx.nd.array([train_loss])
            hvd.allreduce_(train_loss_nd, name='train_loss', average=True)
            train_loss = np.asscalar(train_loss_nd.asnumpy())
            acc_nd = mx.nd.array([acc])
            hvd.allreduce_(acc_nd, name='acc', average=True)
            acc = np.asscalar(acc_nd.asnumpy())
            val_acc_nd = mx.nd.array([val_acc])
            hvd.allreduce_(val_acc_nd, name='val_acc', average=True)
            mx.nd.waitall()
            val_acc = np.asscalar(val_acc_nd.asnumpy())

            if val_acc > best_val_score:
                best_val_score = val_acc
                # net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

            if rank == 0:
                logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f, bits: %f' %
                    (epoch, acc, val_acc, train_loss, time.time()-tic, trainer._optimizer.bit_counter / (1e8)))

                if save_period and save_dir and (epoch + 1) % save_period == 0:
                    net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

        if rank == 0:
            if save_period and save_dir:
                net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))



    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
