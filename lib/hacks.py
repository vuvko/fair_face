import mxnet as mx
from mxnet import gluon


def load_export(model_name, epoch_num):
    sym = mx.sym.load(f'{model_name}-symbol.json')
    net = gluon.SymbolBlock([sym], [mx.sym.var('data')])
    weights_path = f'{model_name}-{epoch_num:04d}.params'
    net.load_parameters(weights_path, ctx=mx.cpu(), cast_dtype=True,
                        allow_missing=True, ignore_extra=True) 
    net.initialize(mx.init.Normal(), ctx=mx.cpu()) 
    net.collect_params().reset_ctx(mx.cpu())
    net.hybridize() 
    net.export('model', epoch_num) 
