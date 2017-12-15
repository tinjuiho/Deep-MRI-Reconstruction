import lasagne
import cascadenet.network.layers as l

from lasagne.layers import batch_norm 
from lasagne.layers import TransposedConv2DLayer as transConv

from collections import OrderedDict


def cascade_resnet_modified(pr, net, input_layer, n=5, nf=64, b=lasagne.init.Constant, **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    net[pr+'conv1'] = l.Conv(input_layer, nf, 3, b=b(), name=pr+'conv1')
    net[pr+'conv2'] = l.Conv(net[pr+'conv1'], nf, 3, b=b(), name=pr+'conv2')
    net[pr+'conv3'] = l.Conv(net[pr+'conv2'], nf, 3, b=b(), name=pr+'conv3')
    net[pr+'conv4'] = l.Conv(net[pr+'conv3'], nf, 3, b=b(), name=pr+'conv4')
    net[pr+'transConv1'] = transConv(net[pr+'conv4'], net[pr+'conv4'].input_shape[1], net[pr+'conv4'].filter_size, stride=net[pr+'conv4'].stride, crop=net[pr+'conv4'].pad, W=net[pr+'conv4'].W, flip_filters=not net[pr+'conv4'].flip_filters)
    net[pr+'transConv2'] = transConv(net[pr+'conv3'], net[pr+'conv3'].input_shape[1], net[pr+'conv3'].filter_size, stride=net[pr+'conv3'].stride, crop=net[pr+'conv3'].pad, W=net[pr+'conv3'].W, flip_filters=not net[pr+'conv3'].flip_filters)
    net[pr+'transConv3'] = transConv(net[pr+'conv2'], net[pr+'conv2'].input_shape[1], net[pr+'conv2'].filter_size, stride=net[pr+'conv2'].stride, crop=net[pr+'conv2'].pad, W=net[pr+'conv2'].W, flip_filters=not net[pr+'conv2'].flip_filters)
    net[pr+'transConv4'] = transConv(net[pr+'conv1'], net[pr+'conv1'].input_shape[1], net[pr+'conv1'].filter_size, stride=net[pr+'conv1'].stride, crop=net[pr+'conv1'].pad, W=net[pr+'conv1'].W, flip_filters=not net[pr+'conv1'].flip_filters)

    net[pr+'conv_aggr'] = l.ConvAggr(net[pr+'conv4'], n_channel, 3, b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer], name=pr+'res')
    output_layer = net[pr+'res']

    return net, output_layer


def cascade_resnet(pr, net, input_layer, n=5, nf=64, b=lasagne.init.Constant, **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    net[pr+'conv1'] = l.Conv(input_layer, nf, 3, b=b(), name=pr+'conv1')

    for i in xrange(2, n):
        net[pr+'conv%d'%i] = l.Conv(net[pr+'conv%d'%(i-1)], nf, 3, b=b(),
                                    name=pr+'conv%d'%i)

    net[pr+'conv_aggr'] = l.ConvAggr(net[pr+'conv%d'%(n-1)], n_channel, 3,
                                     b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
                                    name=pr+'res')
    output_layer = net[pr+'res']
    return net, output_layer


def cascade_resnet_3d_avg(pr, net, input_layer, n=5, nf=64,
                          b=lasagne.init.Constant, frame_dist=range(5),
                          **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    divide_by_n = kwargs['cascade_i'] != 0
    k = (3, 3, 3)

    # Data sharing layer
    net[pr+'kavg'] = l.AverageInKspaceLayer([input_layer, net['mask']],
                                            shape,
                                            frame_dist=frame_dist,
                                            divide_by_n=divide_by_n,
                                            clipped=False)
    # Conv layers
    net[pr+'conv1'] = l.Conv3D(net[pr+'kavg'], nf, k, b=b(), name=pr+'conv1')

    for i in xrange(2, n):
        net[pr+'conv%d'%i] = l.Conv3D(net[pr+'conv%d'%(i-1)], nf, k, b=b(),
                                      name=pr+'conv%d'%i)

    net[pr+'conv_aggr'] = l.Conv3DAggr(net[pr+'conv%d'%(n-1)], n_channel, k,
                                       b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
                                    name=pr+'res')
    output_layer = net[pr+'res']
    return net, output_layer


def build_cascade_cnn_from_list(shape, net_meta, lmda=None):
    """
    Create iterative network with more flexibility

    net_meta: [(model1, cascade1_n),(model2, cascade2_n),....(modelm, cascadem_n),]
    """
    if not net_meta:
        raise

    net = OrderedDict()
    input_layer, kspace_input_layer, mask_layer = l.get_dc_input_layers(shape)
    net['input'] = input_layer
    net['kspace_input'] = kspace_input_layer
    net['mask'] = mask_layer

    j = 0
    for cascade_net, cascade_n in net_meta:
        # Cascade layer
        for i in xrange(cascade_n):
            pr = 'c%d_' % j
            net, output_layer = cascade_net(pr, net, input_layer,
                                            **{'cascade_i': j})

            # add data consistency layer
            net[pr+'dc'] = l.DCLayer([output_layer,
                                      net['mask'],
                                      net['kspace_input']],
                                     shape,
                                     inv_noise_level=lmda)
            input_layer = net[pr+'dc']
            j += 1

    output_layer = input_layer
    return net, output_layer


def build_d2_c2(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_resnet(pr, net, input_layer, n=2)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 2)])


def build_d5_c5(shape):
    # return build_cascade_cnn_from_list(shape, [(cascade_resnet, 5)])
    return build_cascade_cnn_from_list(shape, [(cascade_resnet_modified, 5)])

def build_d2_c2_s(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_resnet_3d_avg(pr, net, input_layer, n=2, nf=16,
                                     frame_dist=range(2), **kwargs)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 2)])


def build_d5_c10_s(shape):
    return build_cascade_cnn_from_list(shape, [(cascade_resnet_3d_avg, 10)])
