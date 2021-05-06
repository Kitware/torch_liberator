"""
Code to load pretrained weights into a model
"""
import numpy as np
import ubelt as ub
import torch
from os.path import exists
from os.path import join
import warnings


def load_partial_state(model, model_state_dict, leftover=None,
                       ignore_unset=False, verbose=2,
                       mangle=True, association=None,
                       initializer=None):
    """
    Load as much state from a state dict into a model

    Args:
        model (torch.nn.Module): module to initialize

        model_state_dict (dict): state dict we wish to transfer

        leftover (callable): fallback method for initializing incompatible
             areas, if none then those areas are left as-is.

        association (str): controls how we search for the association between
            the two model states. Can be strict, module-hack, prefix-hack,
            embedding, or isomorphism.  Default is: module-hack.  I recommend
            trying isomorphism to see if it works for you.

        mangle (bool, default=True): If True, mangles tensors that have the
            same key, but different shapes forcing them to fit. This might
            destroy information when forcing a a larger tensor into a smaller
            tensor, or leave extra uninitialized room when a small tensor is
            placed in a larger one. Note be careful when mangling a
            classification layer if class indexes are not aligned.

        verbose (int): verbosity level

    Returns:
        Dict: info - summary of actions taken

    TODO:
        - [ ] Allow user to specify how incompatible layers are handled.

    Notes:

        Have you ever had the scenario where

        Has anyone ever had a problem where you had a torch model with a state
        dict with keys that looked like: `mymodel.detector.layer1.conv.weight`,
        but you had a pretrained weight file with keys that looked like:
        `module.layer1.conv.weight`?

        The latest version of `torch_liberator.load_patial_state` can handle
        this by solving a maximum-common-subtree-isomorphism problem. This
        computes the largest possible mapping between the two state
        dictionaries that share consistent suffixes.

        >>> # This means you can load an off-the-shelf unmodified pretrained resnet50
        >>> # where the keys might look something like this:
        >>> resnet_keys = {
        >>>     'conv1.weight',
        >>>     'layer1.0.conv1.weight',
        >>>     'layer1.0.conv2.weight',
        >>>     'layer1.0.conv3.weight',
        >>>     'layer1.0.downsample.0.weight',
        >>>     'layer2.0.conv1.weight',
        >>>     'layer2.0.conv2.weight',
        >>>     'layer2.0.conv3.weight',
        >>>     'layer3.0.conv1.weight',
        >>>     'layer4.0.conv1.weight',
        >>>     'fc.weight',
        >>>     'fc.bias',
        >>> }
        >>> #
        >>> # And perhaps you have a model that has a state dict where keys
        >>> # look like this:
        >>> model_keys = {
        >>>     'preproc.conv1.weight'
        >>>     'backbone.layer1.0.conv1.weight',
        >>>     'backbone.layer1.0.conv2.weight',
        >>>     'backbone.layer1.0.conv3.weight',
        >>>     'backbone.layer1.0.downsample.0.weight',
        >>>     'backbone.layer2.0.conv1.weight',
        >>>     'backbone.layer2.0.conv2.weight',
        >>>     'backbone.layer2.0.conv3.weight',
        >>>     'backbone.layer3.0.conv1.weight',
        >>>     'backbone.layer4.0.conv1.weight',
        >>>     'head.conv1'
        >>>     'head.conv2'
        >>>     'head.fc.weight'
        >>>     'head.fc.bias'
        >>> }
        >>> #
        >>> # We can compute a partial mapping between them
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(resnet_keys, model_keys)
        >>> print(ub.repr2(ub.dzip(subpaths1, subpaths2)))
        {
            'layer1.0.conv2.weight':        'backbone.layer1.0.conv2.weight',
            'layer1.0.conv3.weight':        'backbone.layer1.0.conv3.weight',
            'layer1.0.downsample.0.weight': 'backbone.layer1.0.downsample.0.weight',
            'layer2.0.conv1.weight':        'backbone.layer2.0.conv1.weight',
            'layer2.0.conv2.weight':        'backbone.layer2.0.conv2.weight',
            'layer2.0.conv3.weight':        'backbone.layer2.0.conv3.weight',
            'layer3.0.conv1.weight':        'backbone.layer3.0.conv1.weight',
            'layer4.0.conv1.weight':        'backbone.layer4.0.conv1.weight',
        }

        Also, if the sizes of the tensor don't quite fit, they will be
        mangled, i.e. "shoved-in" as best as possible.

    Example:
        >>> import torchvision
        >>> import torch
        >>> resnet50 = torchvision.models.resnet50()
        >>> class CustomModel(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.module = resnet50
        >>>         self.extra = torch.nn.Linear(1, 1)
        >>> model = CustomModel()
        >>> model_state_dict = resnet50.state_dict()
        >>> model_state_dict2 = {'prefix.' + k: v for k, v in model_state_dict.items()}
        >>> import ubelt as ub
        >>> with ub.Timer(verbose=2, label='strict'):
        >>>     load_partial_state(model, model_state_dict, association='strict', verbose=0)
        >>> with ub.Timer(verbose=2, label='prefix-hack'):
        >>>     load_partial_state(model, model_state_dict, association='prefix-hack', verbose=0)
        >>> with ub.Timer(verbose=2, label='module-hack'):
        >>>     load_partial_state(model, model_state_dict, association='module-hack', verbose=0)
        >>> with ub.Timer(verbose=2, label='embedding'):
        >>>     load_partial_state(model, model_state_dict, association='embedding', verbose=0)

        >>> load_partial_state(model, model_state_dict, association='prefix-hack', verbose=1)
        >>> load_partial_state(model, model_state_dict, association='module-hack', verbose=1)
    """
    if association is None:
        association = 'module-hack'  # old default
        # association = 'prefix-hack'  # new default

    if initializer is not None:
        warnings.warn('initializer is deprecated use leftover')
        leftover = initializer

    self_state = model.state_dict()

    def _fix_keys(model_state_dict):
        """
        Hack around DataParallel wrapper. If there is nothing in common between
        the two models check to see if prepending 'module.' to other keys fixes
        it.
        """
        other_keys = set(model_state_dict)
        self_keys = set(self_state)

        if 0:
            # Automatic way to reduce nodes in the trees?
            # If node b always follows node a, can we contract it?
            nodes1 = [n for p in other_keys for n in p.split('.')]
            nodes2 = [n for p in self_keys for n in p.split('.')]
            tups1 = list(tup for key in other_keys for tup in ub.iter_window(key.split('.'), 2))
            tups2 = list(tup for key in self_keys for tup in ub.iter_window(key.split('.'), 2))
            x = ub.ddict(list)
            for a, b in tups1:
                x[a].append(b)
            for a, b in tups2:
                x[a].append(b)

            nodehist = ub.dict_hist(nodes1 + nodes2)

            for k, v in x.items():
                print('----')
                print(k)
                print(nodehist[k])
                follow_hist = ub.dict_hist(v)
                print(follow_hist)
                total = sum(follow_hist.values())
                if ub.allsame(follow_hist.values()) and total == nodehist[k]:
                    print('CONTRACT')

            # pair_freq = ub.dict_hist(ub.flatten([tups1, tups2]))
            # print(forest_str(paths_to_otree(other_keys, '.')))

        # common_keys = other_keys.intersection(self_keys)
        # if not common_keys:
        if not other_keys.issubset(self_keys):
            if association == 'strict':
                pass
            elif association == 'module-hack':
                # If there are no common keys try a hack
                prefix = 'module.'
                def smap(f, ss):
                    return set(map(f, ss))
                def fix1(k):
                    return prefix + k
                def fix2(k):
                    if k.startswith(prefix):
                        return k[len(prefix):]
                if smap(fix1, other_keys).intersection(self_keys):
                    model_state_dict = ub.map_keys(fix1, model_state_dict)
                elif smap(fix2, other_keys).intersection(self_keys):
                    model_state_dict = ub.map_keys(fix2, model_state_dict)
            elif association == 'prefix-hack':
                import functools
                def add_prefix(k, prefix):
                    return prefix + k
                def remove_prefix(k, prefix):
                    if k.startswith(prefix):
                        return k[len(prefix):]
                # set1 = other_keys
                # target_set2 = self_keys
                found = _best_prefix_transform(other_keys, self_keys)
                if found is not None:
                    for action, prefix in found['transform']:
                        if action == 'add':
                            func = functools.partial(add_prefix, prefix=prefix)
                        elif action == 'remove':
                            func = functools.partial(remove_prefix, prefix=prefix)
                        else:
                            raise AssertionError
                        model_state_dict = ub.map_keys(func, model_state_dict)
            elif association in {'embedding', 'isomorphism'}:
                if verbose > 1:
                    print('Using subpath {} association, may take some time'.format(association))
                # I believe this is the correct way to solve the problem
                paths1 = sorted(other_keys)
                paths2 = sorted(self_state)

                if 1:
                    # hack to filter to reduce tree size in embedding problem
                    def shrink_paths(paths):
                        new_paths = []
                        for p in paths:
                            p = p.replace('.0', ':0')
                            p = p.replace('.1', ':1')
                            p = p.replace('.2', ':2')
                            p = p.replace('.3', ':3')
                            p = p.replace('.4', ':4')
                            p = p.replace('.5', ':5')
                            p = p.replace('.6', ':6')
                            p = p.replace('.7', ':7')
                            p = p.replace('.8', ':8')
                            p = p.replace('.9', ':9')
                            p = p.replace('.weight', ':weight')
                            p = p.replace('.bias', ':bias')
                            p = p.replace('.num_batches_tracked', ':num_batches_tracked')
                            p = p.replace('.running_mean', ':running_mean')
                            p = p.replace('.running_var', ':running_var')
                            # p = p.replace('.conv1', ':conv1')
                            # p = p.replace('.conv2', ':conv2')
                            # p = p.replace('.conv3', ':conv3')
                            # p = p.replace('.bn1', ':bn1')
                            # p = p.replace('.bn2', ':bn2')
                            # p = p.replace('.bn3', ':bn3')
                            new_paths.append(p)
                        return new_paths

                    # Reducing the depth saves a lot of time
                    paths1_ = shrink_paths(paths1)
                    paths2_ = shrink_paths(paths2)

                subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1_, paths2_, sep='.', mode=association)
                subpaths1 = [p.replace(':', '.') for p in subpaths1]
                subpaths2 = [p.replace(':', '.') for p in subpaths2]
                mapping = ub.dzip(subpaths1, subpaths2)
                if verbose > 1:
                    other_unmapped = sorted(other_keys - set(mapping.keys()))
                    self_unmapped = sorted(self_keys - set(mapping.values()))
                    print('-- embed association (other -> self) --')
                    print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
                    print('self_unmapped = {}'.format(ub.repr2(self_unmapped, nl=1)))
                    print('other_unmapped = {}'.format(ub.repr2(other_unmapped, nl=1)))
                    print('len(mapping) = {}'.format(ub.repr2(len(mapping), nl=1)))
                    print('len(self_unmapped) = {}'.format(ub.repr2(len(self_unmapped), nl=1)))
                    print('len(other_unmapped) = {}'.format(ub.repr2(len(other_unmapped), nl=1)))
                    print('-- end embed association --')

                # HACK: something might be wrong, there was an instance with
                # HRNet_w32 where multiple keys mapped to the same key
                # bad keys were incre_modules.3.0.conv1.weight and conv1.weight
                #
                # This will not error, but may produce bad output
                try:
                    model_state_dict = ub.map_keys(
                        lambda k: mapping.get(k, k), model_state_dict)
                except Exception as ex:
                    HACK = 1
                    if HACK:
                        new_state_dict_ = {}
                        for k, v in model_state_dict.items():
                            new_state_dict_[mapping.get(k, k)] = v
                        model_state_dict = new_state_dict_
                        warnings.warn('ex = {!r}'.format(ex))
                    else:
                        raise
            else:
                raise KeyError(association)
        return model_state_dict

    other_state = _fix_keys(model_state_dict)

    self_unset_keys = set(self_state.keys())  # will end up as keys in our that were not set
    other_unused_keys = set(other_state.keys())  # will end up as keys in the other model that were not used

    seen_keys = ub.ddict(set)

    for key, other_value in other_state.items():
        if key not in self_state:
            if verbose > 0:
                print('Skipping {} because it does not exist'.format(key))
            seen_keys['skipped'].add(key)
        else:
            self_value = self_state[key]
            if other_value.size() == self_value.size():
                self_state[key] = other_value
                self_unset_keys.remove(key)
                other_unused_keys.remove(key)
                seen_keys['full_add'].add(key)
            elif len(other_value.size()) == len(self_value.size()):
                if not mangle:
                    if verbose > 0:
                        print('Skipping {} due to incompatable size and mangle=False'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                    seen_keys['skipped'].add(key)
                elif key.endswith('bias'):
                    if verbose > 0:
                        print('Skipping {} due to incompatable size'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                    seen_keys['skipped'].add(key)
                else:
                    if leftover is None:
                        if verbose > 0:
                            print('Skipping {} due to incompatable size and no default initializer'.format(key))
                            print(' * self  = {!r}'.format(self_value.size()))
                            print(' * other = {!r}'.format(other_value.size()))
                        seen_keys['skipped'].add(key)
                    else:
                        if verbose > 0:
                            print('Partially add {} with incompatable size'.format(key))
                            print(' * self  = {!r}'.format(self_value.size()))
                            print(' * other = {!r}'.format(other_value.size()))
                        # Initialize all weights in case any are unspecified
                        if leftover is None:
                            try:
                                leftover(self_state[key])
                            except Exception:
                                if verbose > 0:
                                    print('Unable to init {} with {}'.format(key, leftover))

                        # Transfer as much as possible
                        min_size = np.minimum(self_state[key].shape,
                                              other_value.shape)
                        sl = tuple([slice(0, s) for s in min_size])
                        self_state[key][sl] = other_value[sl]

                        # if shock_partial:
                        #     # Shock weights because we are doing something weird
                        #     # might help the network recover in case this is
                        #     # not a good idea
                        #     shock(self_state[key], func=leftover)
                        self_unset_keys.remove(key)
                        other_unused_keys.remove(key)

                        if self_state[key].numel() < other_value.numel():
                            seen_keys['partial_add_some'].add(key)
                        else:
                            seen_keys['partial_add_all'].add(key)
            else:
                if verbose > 0:
                    print('Skipping {} due to incompatable size'.format(key))
                    print(' * self  = {!r}'.format(self_value.size()))
                    print(' * other = {!r}'.format(other_value.size()))
                seen_keys['skipped'].add(key)

    if ignore_unset is True:
        self_unset_keys = []
    elif ignore_unset:
        self_unset_keys = list(ub.oset(self_unset_keys) - set(ignore_unset))

    if (self_unset_keys or other_unused_keys or
         seen_keys['partial_add_some'] or seen_keys['partial_add_all']):
        if verbose > 0:
            if seen_keys:
                print('Pretrained weights are a partial fit')
            else:
                print('Pretrained weights do not fit!')
        if verbose > 1:
            print('Seen Keys: {}'.format(ub.repr2(seen_keys, nl=2)))
            print('Self Unset Keys: {}'.format(ub.repr2(self_unset_keys, nl=1)))
            print('Other Unused keys: {}'.format(ub.repr2(other_unused_keys, nl=1)))
            print('summary:')
            seen_sum = ub.map_vals(len, seen_keys)
            print('Seen Num: {}'.format(ub.repr2(seen_sum, nl=2)))
            print('Self Unset Num: {}'.format(ub.repr2(len(self_unset_keys), nl=1)))
            print('Other Unused Num: {}'.format(ub.repr2(len(other_unused_keys), nl=1)))
        if leftover:
            if verbose > 0:
                print('Initializing unused keys using {}'.format(leftover))
            for key in self_unset_keys:
                if key.endswith('.num_batches_tracked'):
                    pass  # ignore num_batches_tracked
                elif key.endswith('.bias'):
                    self_state[key].fill_(0)
                else:
                    try:
                        leftover(self_state[key])
                    except Exception:
                        if verbose > 0:
                            print('Unable to init {} with {}'.format(key, leftover))

    else:
        if verbose > 0:
            print('Pretrained weights are a perfect fit')
    model.load_state_dict(self_state)

    info = {
        'seen': seen_keys,
        'self_unset': self_unset_keys,
        'other_unused': other_unused_keys
    }
    return info


class Pretrained(object):
    """
    This class is a stub version of netharn.initializers.Pretrained that is
    with only the functionality needed by torch_liberator.

    Attributes:
        fpath (str | PathLike): location of the pretrained weights file.

            This can be a pytorch '.pt' file containing the model state, a path
            to a deploy '.zip' file.

            While it is best practice to use an explicit filepath, we do allow
            `fpath` be a "fuzzy" glob string as long as the pattern resolves to
            a single file, otherwise an error will be thrown.
    """
    def __init__(self, fpath):
        self.fpath = fpath

    def __nice__(self):
        return self.fpath

    def _rectify_deploy_zip_weights_path(self):
        # Find the path to the weights inside the zipfile
        import zipfile
        fpath = None
        candidates = []
        with zipfile.ZipFile(self.fpath, 'r') as myzip:
            for zinfo in myzip.filelist:
                if zinfo.filename.endswith('deploy_snapshot.pt'):
                    candidates = [zinfo.filename]
                    break
                elif zinfo.filename.endswith('.pt'):
                    candidates.append(zinfo.filename)
        if len(candidates) == 0:
            raise OSError('Cannot find pretrained weights in {}'.format(
                self.fpath))
        elif len(candidates) > 1:
            raise OSError('Multiple weights files in {}'.format(
                self.fpath))
        else:
            fpath = join(self.fpath, candidates[0])
        return fpath

    def _rectify_fpath(self):
        """
        Resolves the `self.fpath`, which may be non-physical path (e.g.
        globstring or zipfile) to an existing physical path if possible.
        """
        if self.fpath is None:
            raise ValueError('Pretrained fpath is None!')
        # Handle torch deployment zipfiles
        if exists(self.fpath) and self.fpath.endswith('.zip'):
            fpath = self._rectify_deploy_zip_weights_path()
        else:
            fpath = self.fpath
            if not exists(fpath) and '*' in fpath:
                import glob
                cands = list(glob.glob(fpath))
                if len(cands) == 1:
                    fpath = cands[0]
                else:
                    raise Exception(
                        'Pattern fpath={!r} must resolve to exactly one file, '
                        'but got cands{!r}'.format(fpath, cands))
        return fpath

    def forward(self, model, verbose=2):
        """
        Apply the pretrained weights to the model
        """
        from torch_liberator import util

        main_device_id = _main_device_id_from_data(model)
        fpath = self._rectify_fpath()
        try:
            file = util.zopen(fpath, 'rb', seekable=True)
            model_state_dict = _torch_load(file, main_device_id)
        except Exception:
            print('Failed to open fpath = {!r}'.format(fpath))
            raise

        if 'model_state_dict' in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
        elif 'state_dict' in model_state_dict:
            model_state_dict = model_state_dict['state_dict']
        elif 'weights' in model_state_dict:
            model_state_dict = model_state_dict['weights']
        else:
            # If the dictionary is flat (i.e. all values are tensors) then it
            # is safe to assume this file only contains weights.
            # Otherwise raise an exception.
            if not all(torch.is_tensor(v) for v in model_state_dict.values()):
                raise Exception(
                    'snapshot file is nested, but does not have expected keys: '
                    'model_state_dict or weights. Root keys are {}'.format(
                        sorted(model_state_dict.keys())
                    ))
        # Remove any DataParallel / DataSerial
        raw_model = _raw_model(model)
        info = load_partial_state(raw_model, model_state_dict, leftover=None,
                                  mangle=False, verbose=verbose)
        return info


def _main_device_id_from_data(item):
    """
    Get device ids of a model

    Example:
        >>> device_ids = _main_device_id_from_data(torch.randn(3))
        >>> print('device_ids = {!r}'.format(device_ids))
        >>> if torch.cuda.is_available():
        >>>     device_ids = _main_device_id_from_data(torch.randn(3).to('cuda'))
        >>>     print('device_ids = {!r}'.format(device_ids))
        >>>     for i in range(torch.cuda.device_count()):
        >>>         device_ids = _main_device_id_from_data(torch.randn(3).to(i))
        >>>         print('device_ids = {!r}'.format(device_ids))
    """
    if hasattr(item, 'device'):
        return item.device.index
    if hasattr(item, 'is_cuda'):
        if item.is_cuda:
            return item.get_device().index
        else:
            return None
    elif hasattr(item, 'state_dict'):
        devices = [item.device for item in item.state_dict().values()]
        _device_ids = set()
        for device in devices:
            if device.type == 'cuda':
                index = device.index or 0
                _device_ids.add(index)
            else:
                _device_ids.add(None)
        try:
            _device_ids = sorted(_device_ids)
        except TypeError:
            raise Exception('cannot currently mix CPU and GPU')
        _device_id = ub.peek(_device_ids)
        return _device_id
    else:
        raise TypeError(type(item))


def _raw_model(model):
    """
    Unmounts the original core model if it is mounted.

    Args:
        model (torch.nn.Module): a model (potentially mounted)

    Returns:
        torch.nn.Module:
            if `model` is mounted returns `model.module`
            otherwise, returns `model`
    """
    if hasattr(model, 'module'):
        # hack, not as safe as the netharn version
        model = model.module
    return model


def _torch_load(fpath, main_device_id=None):
    """
    Loads data from a filepath onto a device

    Args:
        fpath (str or file): path to torch data file or file-like object
    """
    def _map_location(storage, location):
        """
        Helper for torch.load

        Args:
            storage (torch.Storage) : the initial deserialization of the
                storage of the data read by `torch.load`, residing on the CPU.
            location (str): tag identifiying the location the data being read
                by `torch.load` was originally saved from.

        Returns:
            torch.Storage : the storage
        """
        if main_device_id is None:
            return storage
        else:
            return storage.cuda(main_device_id)
    print('Loading data onto {} from {}'.format(main_device_id, fpath))
    try:
        return torch.load(fpath, map_location=_map_location)
    except Exception:
        print('Failed to load fpath={}'.format(fpath))
        raise


def _best_prefix_transform(set1, target_set2):
    """
    Find a way to transform prefixes of items in set1 to match target_set2

    Example:
        >>> set1 = {'mod.f.0.w',
        >>>         'mod.f.1.b',
        >>>         'mod.f.1.n',
        >>>         'mod.f.1.rm',
        >>>         'mod.f.1.rv',}
        >>> #
        >>> target_set2 = {
        >>>      'bar.foo.extra.f.1.b',
        >>>      'bar.foo.extra.f.1.n',
        >>>      'bar.foo.extra.f.1.w',
        >>>      'bar.foo.extra.f.3.w',
        >>> }
        >>> _best_prefix_transform(set1, target_set2)
        >>> target_set2.add('JUNK')
        >>> _best_prefix_transform(set1, target_set2)
    """

    # probably an efficient way to do this with a trie

    # NOTE: In general this is a graph-isomorphism problem or a  maximum common
    # subgraph problem. However, we can look only at the special case of
    # "maximum common subtrees". Given two directory structures (as trees)
    # we find the common bits.
    # https://perso.ensta-paris.fr/~diam/ro/online/viggo_wwwcompendium/node168.html
    # We can approximate to O(log log n / log^2 n)
    # Can get algorithm from maximum independent set
    # https://arxiv.org/abs/1602.07210

    # The most efficient algorithm here would be for solving
    # "Maximum common labeled subtrees"
    # APX-hard for unordered trees, but polytime solveable for ordered trees
    # For directory structures we can induce an order, and hense obtain a
    # polytime solution
    # #
    # On the Maximum Common Embedded Subtree Problem for Ordered Trees
    # https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf

    from os.path import commonprefix
    prefixes1 = commonprefix(list(set1)).split('.')
    prefixes2 = commonprefix(list(target_set2)).split('.')

    # Remove the trailing prefixes that are the same
    num_same = 0
    for i in range(1, min(len(prefixes1), len(prefixes2))):
        if prefixes1[-i] == prefixes2[-i]:
            num_same = i
        else:
            break
    prefixes1 = prefixes1[:-num_same]
    prefixes2 = prefixes2[:-num_same]

    ALLOW_FUZZY = 1
    if ALLOW_FUZZY and len(prefixes2) == 0:
        # SUPER HACK FOR CASE WHERE THERE IS JUST ONE SPOILER ELEMENT IN THE
        # TARGET SET. THE ALGORITHM NEEDS TO BE RETHOUGHT FOR THAT CASE
        possible_prefixes = [k.split('.') for k in target_set2]
        prefix_hist = ub.ddict(lambda: 0)
        for item in possible_prefixes:
            for i in range(1, len(item)):
                prefix_hist[tuple(item[0:i])] += 1
        prefixes2 = ['.'.join(ub.argmax(prefix_hist))]

    def add_prefix(items, prefix):
        return {prefix + k for k in items}
    def remove_prefix(items, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k for k in items}

    import itertools as it
    found_cand = []
    for i1, i2 in it.product(range(len(prefixes1) + 1), range(len(prefixes2) + 1)):
        if i1 == 0 and i2 == 0:
            continue
        # Very inefficient, we should be able to do better
        prefix1 = '.'.join(prefixes1[:i1])
        prefix2 = '.'.join(prefixes2[:i2])
        if prefix1:
            prefix1 = prefix1 + '.'
        if prefix2:
            prefix2 = prefix2 + '.'

        # We are allowed to remove a prefix from a set, add the other
        # prefix to the set, or remove and then add.
        set1_cand1 = remove_prefix(set1, prefix1)
        set1_cand2 = add_prefix(set1, prefix2)
        set1_cand3 = add_prefix(set1_cand1, prefix2)

        common1 = set1_cand1 & target_set2
        common2 = set1_cand2 & target_set2
        common3 = set1_cand3 & target_set2
        if common1:
            found_cand.append({
                'transform': [('remove', prefix1)],
                'value': len(common1),
            })
        if common2:
            found_cand.append({
                'transform': [('add', prefix2)],
                'value': len(common2),
            })
        if common3:
            found_cand.append({
                'transform': [('remove', prefix1), ('add', prefix2)],
                'value': len(common3),
            })
    if len(found_cand):
        found = max(found_cand, key=lambda x: x['value'])
    else:
        found = None
    return found


def maximum_common_ordered_subpaths(paths1, paths2, sep='.', mode='embedding'):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/functional.py maximum_common_ordered_subpaths:0 --profile && cat profile_output.txt
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/functional.py maximum_common_ordered_subpaths:0

    Example:
        >>> import torchvision
        >>> resnet50 = torchvision.models.resnet50()
        >>> paths1 = sorted(resnet50.state_dict().keys())
        >>> paths2 = ['prefix.' + k for k in paths1]
        >>> paths2.append('extra_key')
        >>> sep = '.'
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep, mode='embedding')
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('embedding mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep, mode='isomorphism')
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('isomorphism mapping = {}'.format(ub.repr2(mapping, nl=1)))

        if 0:
            import timerit
            ti = timerit.Timerit(2, bestof=2, verbose=2)
            for timer in ti.reset('embedding'):
                with timer:
                    maximum_common_ordered_subpaths(paths1, paths2, mode='embedding')
            for timer in ti.reset('isomorphism'):
                with timer:
                    maximum_common_ordered_subpaths(paths1, paths2, mode='isomorphism')

    Example:
        >>> rng = None
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(rng)
        >>> def random_paths(rng, max_depth=10):
        >>>     depth = rng.randint(1, max_depth)
        >>>     parts = list(map(chr, rng.randint(ord('a'), ord('z'), size=depth)))
        >>>     path = '.'.join(parts)
        >>>     return path
        >>> n = 50
        >>> paths1 = sorted({random_paths(rng) for _ in range(n)})
        >>> paths2 = sorted({random_paths(rng) for _ in range(n)})
        >>> paths1 = paths1 + ['a.' + k for k in paths2[0:n // 3]]
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))

    Example:
        >>> paths1 = [
        >>>     'stats',
        >>>     'z.mod.f.0.w',
        >>>     'a.z.mod.f.0.b',
        >>>     'z.mod.f.1.b',
        >>>     'z.mod.f.1.n',
        >>>     'z.mod.f.1.m',
        >>>     'z.mod.f.1.v',
        >>>     'z.mod.f.2.m',
        >>>     'z.mod.z.q'
        >>> ]
        >>> # paths1 = ['mod']
        >>> #
        >>> paths2 = [
        >>>     'stats',
        >>>     'bar.f.0.w',
        >>>     'bar.foo.extra.z.q',
        >>>     'bar.foo.extra',
        >>>     'bar.foo.extra.f.1.b',
        >>>     'bar.foo.extra.f.1.n',
        >>>     'bar.foo.extra.f.1.w',
        >>>     'bar.foo.extra.f.3.z',  # FIXME we need to handle label comparision operators
        >>>     # I think we allow labels to match if they have the same suffix
        >>> ]
        >>> sep = '.'
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep, mode='embedding')
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('embedding mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep, mode='isomorphism')
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('isomorphism mapping = {}'.format(ub.repr2(mapping, nl=1)))


    Example:
        >>> sep = '.'
        >>> paths1 = ['a.b']
        >>> paths2 = ['a.b']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> paths1 = ['c.a.b']
        >>> paths2 = ['a.b']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> paths1 = ['c.a.b', 'c.a.e', 'c.a.q']
        >>> paths2 = ['a.b', 'c.e', 'c.a', 'a.q']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
    """
    import networkx as nx
    from torch_liberator import _nx_ext_v2

    # the longest common balanced sequence problem
    def _affinity(tok1, tok2):
        score = 0
        for t1, t2 in zip(tok1[::-1], tok2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score

        # return tok1[-1] == tok2[-1]
    node_affinity = _affinity
    # import operator
    # eq = operator.eq

    def paths_to_otree(paths):
        tree = nx.OrderedDiGraph()
        for path in sorted(paths):
            parts = tuple(path.split(sep))
            node_path = []
            for i in range(1, len(parts) + 1):
                node = parts[0:i]
                tree.add_node(node)
                tree.nodes[node]['label'] = node[-1]
                node_path.append(node)
            for u, v in ub.iter_window(node_path, 2):
                tree.add_edge(u, v)
        return tree

    tree1 = paths_to_otree(paths1)
    tree2 = paths_to_otree(paths2)

    if mode == 'embedding':
        subtree1, subtree2, value = _nx_ext_v2.maximum_common_ordered_subtree_embedding(tree1, tree2, node_affinity=node_affinity)
    elif mode == 'isomorphism':
        subtree1, subtree2, value = _nx_ext_v2.maximum_common_ordered_subtree_isomorphism(tree1, tree2, node_affinity=node_affinity)
    else:
        raise KeyError(mode)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2
