# flake8: noqa


def example1():
    """
    import torchvision
    import torch_liberator
    # Create two similar but different models
    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model = torchvision.models.resnet50(pretrained=True)

    faster_rcnn.load_state_dict(model.state_dict())

    # Return a dictionary that tells what load_partial_state did
    info = torch_liberator.load_partial_state(
         faster_rcnn, model.state_dict(),
         association='isomorphism')

    >>> print(ub.map_vals(len, info['seen']))
    >>> print(ub.map_vals(len, ub.dict_diff(info, ['seen'])))
    {'full_add': 265, 'skipped': 55}
    {'other_unused': 55, 'self_unset': 30}
    """



































def example2():
    """
    >>> # Load partial state return a dictionary that tells you how well it did
    >>> info = torch_liberator.load_partial_state(
    >>>     faster_rcnn, state_dict, verbose=0, association='isomorphism')
    >>> print(ub.map_vals(len, info['seen']))
    >>> print(ub.map_vals(len, ub.dict_diff(info, ['seen'])))
    {'full_add': 265, 'skipped': 55}
    {'other_unused': 55, 'self_unset': 30}


    """


def has_this_ever_happened_to_you():

    import torchvision
    import torch
    raw_model = torchvision.models.resnet18(pretrained=True)

    dp_model = torch.nn.DataParallel(raw_model, device_ids=[0])
    dp_model.load_state_dict(raw_model.state_dict())

    """
    RuntimeError: Error(s) in loading state_dict for DataParallel:
    Missing key(s) in state_dict: "module.conv1.weight",
    """

    # FIX IT
    import torch_liberator
    _ = torch_liberator.load_partial_state(dp_model, raw_model.state_dict(), association='isomorphism')
    """
    Pretrained weights are a perfect fit
    """

    rank = 0
    world_size = 1
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    ddp_model = torch.nn.parallel.distributed.DistributedDataParallel(raw_model, device_ids=[0])
    ddp_model.load_state_dict(raw_model.state_dict())

    """
    RuntimeError: Error(s) in loading state_dict for DistributedDataParallel:
    Missing key(s) in state_dict: "module.conv1.weight", "module.bn1.weight"
    """


def visualization():
    import torchvision
    import torch_liberator
    import networkx as nx
    import kwplot
    import kwimage
    import graphid
    import torch
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    kwplot.autoplt()

    clf_model = torchvision.models.resnet18(pretrained=True)

    clf_model2 = torchvision.models.resnet18(pretrained=False)
    clf_model2.conv1 = torch.nn.Conv2d(13, 64, 7, 2, 3)
    clf_model2.fc = torch.nn.Linear(512, 20)

    det_model = resnet_fpn_backbone('resnet18', pretrained=False)

    wrp_model = torch.nn.DataParallel(clf_model, device_ids=[0])

    def model_to_simplified_paths(model, name='.'):
        paths = [name + '.' + p for p in model.state_dict().keys()]
        paths = [p for p in paths if not p.endswith('num_batches_tracked')]
        paths = [p for p in paths if not p.endswith('running_var')]
        paths = [p for p in paths if not p.endswith('running_mean')]
        paths = [p for p in paths if not p.endswith('bias')]
        paths = [p[:-len('.weight')] if p.endswith('.weight') else p for p in paths]
        # paths = [p.replace('.weight', '') for p in paths]
        paths = [p for p in paths if 'layer2' not in p]
        paths = [p for p in paths if 'layer3' not in p]
        return paths

    clf_paths = model_to_simplified_paths(clf_model, 'clf')
    det_paths = model_to_simplified_paths(det_model, 'det')

    clf_tree = torch_liberator.initializer.paths_to_otree(clf_paths, sep='.')
    det_tree = torch_liberator.initializer.paths_to_otree(det_paths, sep='.')

    clf_tree = torch_liberator.initializer.paths_to_otree(list(clf_model.state_dict().keys()), sep='.')
    det_tree = torch_liberator.initializer.paths_to_otree(list(det_model.state_dict().keys()), sep='.')

    from torch_liberator import _nx_ext_v2
    node_affinity = torch_liberator.initializer._common_suffix_affinity
    subtree1, subtree2, value = _nx_ext_v2.maximum_common_ordered_subtree_isomorphism(
        clf_tree, det_tree, node_affinity=node_affinity)

    wrp_paths = model_to_simplified_paths(wrp_model, 'wrp')
    wrp_tree = torch_liberator.initializer.paths_to_otree(wrp_paths, sep='.')

    nx.set_node_attributes(wrp_tree, name='color', values=kwimage.Color('lightpink').as01())
    nx.set_node_attributes(clf_tree, name='color', values=kwimage.Color('lightblue').as01())
    nx.set_node_attributes(det_tree, name='color', values=kwimage.Color('lightgreen').as01())

    layoutkw = {'rankdir': 'LR'}
    kwplot.figure(fnum=1, doclf=True)
    _ = graphid.util.util_graphviz.show_nx(
        clf_tree,
        fnum=1,
        as_directed=False,
        fontsize=10,
        layoutkw=layoutkw,
    )

    kwplot.figure(fnum=2, doclf=True)
    _ = graphid.util.util_graphviz.show_nx(
        det_tree,
        fnum=2,
        as_directed=False,
        fontsize=10,
        layoutkw=layoutkw,
    )

    kwplot.figure(fnum=4, doclf=True)
    _ = graphid.util.util_graphviz.show_nx(
        wrp_tree,
        fnum=4,
        as_directed=False,
        fontsize=10,
        layoutkw=layoutkw,
    )

    clf_tree = torch_liberator.initializer.paths_to_otree(clf_paths, sep='/')
    det_tree = torch_liberator.initializer.paths_to_otree(det_paths, sep='/')
    nx.set_node_attributes(clf_tree, name='color', values=kwimage.Color('lightblue').as01())
    nx.set_node_attributes(det_tree, name='color', values=kwimage.Color('lightgreen').as01())
    _ = graphid.util.util_graphviz.nx_agraph_layout(clf_tree, inplace=1, **layoutkw)  # NOQA
    _ = graphid.util.util_graphviz.nx_agraph_layout(det_tree, inplace=1, **layoutkw)  # NOQA
    stacked = graphid.util.util_graphviz.stack_graphs([clf_tree, det_tree], vert=False)
    for n1, n2 in zip(subtree1.nodes, subtree2.nodes):
        stacked.add_edge(n1, n2, implicit=True, color=kwimage.Color('orange').as01())
    # stacked.add_edge(('clf',), ('det',), implicit=True)

    # Does only neato respect pin?
    layoutkw_ = layoutkw.copy()
    layoutkw_['prog'] = 'neato'
    _ = graphid.util.util_graphviz.nx_agraph_layout(stacked, inplace=True, **layoutkw_)  # NOQA

    layout_info = graphid.util.util_graphviz.get_nx_layout(stacked, 'custom')
    kwplot.figure(fnum=5).gca().cla()
    _ = graphid.util.util_graphviz.show_nx(
        stacked,
        as_directed=False,
        fnum=3,
        layout='custom',
        fontsize=10,
        layoutkw=layoutkw_, verbose=4)

    # if 0:
    #     agraph = graphid.util.util_graphviz.make_agraph(stacked2.copy())
    #     layoutkw_ = layoutkw.copy()
    #     prog = layoutkw_.pop('prog', 'dot')
    #     argparts = ['-G%s=%s' % (key, str(val))
    #                 for key, val in layoutkw.items()]
    #     args = ' '.join(argparts)
    #     agraph.layout(prog=prog, args=args)
    #     agraph.draw('foo.png')
    #     import xdev
    #     xdev.startfile('foo.png')
