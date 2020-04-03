"""
Code to load pretrained weights into a model
"""
import numpy as np
import ubelt as ub
import torch
from os.path import exists
from os.path import join


def load_partial_state(model, model_state_dict, leftover=None,
                       ignore_unset=False, verbose=2,
                       mangle=True, initializer=None):
    """
    Args:
        model (torch.nn.Module): module to initialize

        model_state_dict (dict): state dict we wish to transfer

        leftover (callable): fallback method for initializing incompatible
             areas, if none then those areas are left as-is.

        mangle (bool, default=True): If True, mangles tensors that have the
            same key, but different shapes forcing them to fit. This might
            destroy information when forcing a a larger tensor into a smaller
            tensor, or leave extra uninitialized room when a small tensor is
            placed in a larger one. Note be careful when mangling a
            classification layer if class indexes are not aligned.

        verbose (int): verbosity level

    Returns:
        Dict: info - summary of actions taken
    """
    if initializer is not None:
        import warnings
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

        if not other_keys.intersection(self_keys):
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
