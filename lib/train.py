import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import gluonfr
from time import time
from tqdm import tqdm
from more_itertools import unzip
from sklearn.metrics import roc_curve, roc_auc_score
from .basic_config import BasicConfig
from .dataset import InfoDataset, PairDataset, ImgRecDataset
from .sampler import UniformClassSampler
from .utils import prepare_experiment, ensure_path
from .data import aggregate_subjects, split_data, sample_pairs


def train(config: BasicConfig, data_df: pd.DataFrame) -> None:
    res = prepare_experiment(config)
    if res is None:
        return None
    experiment_path, logger = res
    use_gpu = len(config.gpus) > 0
    if use_gpu:
        ctx = [mx.gpu(cur_idx) for cur_idx in config.gpus]
    else:
        ctx = [mx.cpu()]
    # train_val_ratio = 0.9
    # subj_dict = aggregate_subjects(list(data_df.index), data_df['SUBJECT_ID'])
    # train_subj, val_subj = split_data(subj_dict, train_val_ratio)
    # train_df = data_df.iloc[sum(list(train_subj.values()), [])]
    # val_pairs = sample_pairs(val_subj, config.val_num_sample)
    # val_idx, val_labels = unzip(val_pairs)
    # val_path_pairs = [(data_df['img_path'][left], data_df['img_path'][right]) for left, right in val_idx]
    train_df = data_df
    dataset = InfoDataset(train_df, filter_fn=config.filter_fn, augs=config.train_augmentations)
    # dataset = ImgRecDataset(config.extra_rec[0], augs=config.train_augmentations)
    train_data = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not config.uniform_subjects,
        sampler=UniformClassSampler(dataset) if config.uniform_subjects else None,
        last_batch='discard',
        num_workers=config.num_workers,
        pin_memory=use_gpu
    )
    # val_dataset = PairDataset(val_path_pairs, val_labels, augs=config.test_augmentations)
    # val_data = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     pin_memory=use_gpu
    # )
    model_name = 'VGG2-ResNet50-Arcface'
    net_name = config.name
    weight_path = str(Path.home() / f'.insightface/models/{model_name}/model-0000.params')
    sym_path = str(Path.home() / f'.insightface/models/{model_name}/model-symbol.json')
    num_subjects = dataset.num_labels
    warmup = config.warmup_epoch * len(train_data)
    lr = config.initial_lr
    cooldown = config.cooldown_epoch * len(train_data)
    lr_factor = config.lr_factor
    num_epoch = config.num_epochs
    momentum = config.momentum
    wd = config.weight_decay
    clip_gradient = config.clip_gradient
    lr_steps = config.steps
    snapshots_path = ensure_path(experiment_path / 'snapshots')
    sym = mx.sym.load(str(sym_path))
    sym = sym.get_internals()['fc1_output']
    if config.normalize:
        norm_sym = mx.sym.sqrt(mx.sym.sum(sym ** 2, axis=1, keepdims=True) + 1e-6)
        sym = mx.sym.broadcast_div(sym, norm_sym, name='fc_normed') * 32
    embeddings = sym
    fc_weights = mx.sym.Variable('fc_weight',
                                 shape=(num_subjects, 512),
                                 init=mx.initializer.Xavier(rnd_type='gaussian',
                                                            factor_type="in",
                                                            magnitude=2),
                                 lr_mult=1)
    if config.weight_normalize:
        fc_weights = mx.sym.L2Normalization(data=fc_weights,
                                            name='norm_fc_weight')
    sym = mx.sym.FullyConnected(sym, weight=fc_weights, num_hidden=num_subjects, name='fc_classification', no_bias=False)
    sym = mx.sym.Group([embeddings, sym])
    net = gluon.SymbolBlock([sym], [mx.sym.var('data')])
    # net.load_parameters(str(weight_path), ctx=mx.cpu(), cast_dtype=True,
    #                     allow_missing=True, ignore_extra=True)
    net.initialize(mx.init.Normal(), ctx=mx.cpu())
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    softmax = gluon.loss.SoftmaxCrossEntropyLoss()
    softmax.hybridize()
    center = gluonfr.loss.CenterLoss(num_subjects, 512, 1e2)
    center.initialize(ctx=mx.cpu())
    center.hybridize()
    center.collect_params().reset_ctx(ctx)
    arc = gluonfr.loss.ArcLoss(num_subjects, m=0.7, s=32, easy_margin=False)

    all_losses = [
        # ('softmax', lambda ots, gts: softmax(ots[1], gts)),
        # ('arc', lambda ots, gts: arc(ots[1], gts)),
        ('center', lambda ots, gts: center(ots[1], gts, ots[0]))
    ]
    # all_losses[1][1].initialize(mx.init.Normal(), ctx=mx.cpu())
    # all_losses[1][1].collect_params().reset_ctx(ctx)
    # [cur_loss[1].hybridize() for cur_loss in all_losses]

    if warmup > 0:
        start_lr = 1e-10
    else:
        start_lr = lr
    warmup_iter = 0
    end_iter = num_epoch * len(train_data)
    cooldown_start = end_iter - cooldown
    cooldown_iter = 0
    end_lr = 1e-10
    param_dict = net.collect_params()
    for key, val in param_dict._params.items():
        if key.startswith('fc_classification'):
            val.lr_mult *= config.classifier_mult
    trainer = mx.gluon.Trainer(param_dict, 'sgd', {
        'learning_rate': start_lr, 'momentum': momentum, 'wd': wd, 'clip_gradient': clip_gradient})

    lr_counter = 0
    num_batch = len(train_data)

    for epoch in range(num_epoch):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * lr_factor)
            lr_counter += 1

        tic = time()
        losses = [0] * len(all_losses)
        metric = mx.metric.Accuracy()
        logger.info(f' > training {epoch}')
        for i, batch in tqdm(enumerate(train_data), total=len(train_data)):
            if warmup_iter < warmup:
                cur_lr = (warmup_iter + 1) * (lr - start_lr) / warmup + start_lr
                trainer.set_learning_rate(cur_lr)
                warmup_iter += 1
            elif cooldown_iter > cooldown_start:
                cur_lr = (end_iter - cooldown_iter) * (trainer.learning_rate - end_lr) / cooldown + end_lr
                trainer.set_learning_rate(cur_lr)
            cooldown_iter += 1
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, even_split=False)
            gts = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, even_split=False)
            with ag.record():
                outputs = [net(X) for X in data]
                if np.any([np.any(np.isnan(o.asnumpy())) for os in outputs for o in os]):
                    print('OOps!')
                    raise RuntimeError
                cur_losses = [[cur_loss(o, l) for (o, l) in zip(outputs, gts)] for _, cur_loss in all_losses]
                metric.update(gts, [ots[1] for ots in outputs])
                combined_losses = [cur[0] for cur in zip(*cur_losses)]
                if np.any([np.any(np.isnan(l.asnumpy())) for l in cur_losses[0]]):
                    print('OOps2!')
                    raise RuntimeError
            for combined_loss in combined_losses:
                combined_loss.backward()

            trainer.step(config.batch_size, ignore_stale_grad=True)
            for idx, cur_loss in enumerate(cur_losses):
                losses[idx] += sum([l.mean().asscalar() for l in cur_loss]) / len(cur_loss)

            if (i + 1) % 1000 == 0:
                # net.save_parameters(str(snapshots_path / f'{net_name}-{(epoch + 1):04d}.params'))
                net.export(str(snapshots_path / f'{net_name}_{i + 1}'), epoch + 1)
                i_losses = [sum([l.mean().asscalar() for l in cur_loss]) / len(cur_loss) for cur_loss in cur_losses]
                losses_str = [f'{l_name}: {i_losses[idx]:.3f}' for idx, (l_name, _) in enumerate(all_losses)]
                losses_str = '; '.join(losses_str)
                m_name, m_val = metric.get()
                losses_str += f'| {m_name}: {m_val}'
                logger.info(f'[Epoch {epoch:03d}][{i+1}] {losses_str} | time: {time() - tic:.1f}')

        if (epoch + 1) % config.save_epoch == 0:
            # net.save_parameters(str(snapshots_path / f'{net_name}-{(epoch + 1):04d}.params'))
            net.export(str(snapshots_path / f'{net_name}'), epoch + 1)

        losses = [l / num_batch for l in losses]
        losses_str = [f'{l_name}: {losses[idx]:.3f}' for idx, (l_name, _) in enumerate(all_losses)]
        losses_str = '; '.join(losses_str)
        m_name, m_val = metric.get()
        losses_str += f'| {m_name}: {m_val}'
        logger.info(f'[Epoch {epoch:03d}] {losses_str} | time: {time() - tic:.1f}')
        # validation
        # logger.info(f' > evaluation {epoch}')
        # for i, batch in tqdm(enumerate(val_data), total=len(val_data)):
        #     left = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, even_split=False)
        #     right = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, even_split=False)
        #     gts = mx.gluon.utils.split_and_load(batch[2], ctx_list=ctx, even_split=False)
