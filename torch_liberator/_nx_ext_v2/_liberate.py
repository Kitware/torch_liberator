import ubelt as ub
from os.path import dirname
import networkx as nx
import liberator


def copy_over_stuff_from_nx_pr():
    """
    Copy from networkx dev/ordered_subtree_isomorphism b08106baae7987af1dc755a6308fcd11bc21cbc8
    """
    dst = ub.expandpath('~/code/netharn/netharn/initializers/_nx_ext_v2')

    from os.path import join
    nx_repo = dirname(dirname(nx.__file__))

    to_copy = [
        join(nx_repo, 'networkx/algorithms/string/_autojit.py'),
        join(nx_repo, 'networkx/algorithms/string/balanced_embedding.py'),
        join(nx_repo, 'networkx/algorithms/string/balanced_embedding_cython.pyx'),
        join(nx_repo, 'networkx/algorithms/string/balanced_isomorphism.py'),
        join(nx_repo, 'networkx/algorithms/string/balanced_isomorphism_cython.pyx'),
        join(nx_repo, 'networkx/algorithms/string/balanced_sequence.py'),
        join(nx_repo, 'networkx/algorithms/minors/tree_embedding.py'),
        join(nx_repo, 'networkx/algorithms/minors/tree_isomorphism.py'),
    ]

    import shutil
    fpath_list = []
    for fpath in to_copy:
        fpath2 = ub.augpath(fpath, dpath=dst)
        fpath_list.append(fpath2)
        shutil.copy2(fpath, fpath2)

    util_fpath = join(dst, 'utils.py')
    closer = liberator.Closer()
    closer.add_dynamic(nx.forest_str)
    closer.add_dynamic(nx.random_ordered_tree)
    closer.add_dynamic(nx.random_tree)
    with open(util_fpath, 'w') as file:
        file.write(closer.current_sourcecode())

    from rob import rob_nav
    force = True
    # force = 0
    rob_nav._ut_sed(r'networkx\.algorithms\.string', 'netharn.initializers._nx_ext_v2', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'networkx/networkx/algorithms/string', 'netharn/netharn/initializers/_nx_ext_v2', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'networkx/algorithms/string', 'netharn/initializers/_nx_ext_v2', fpath_list=fpath_list, force=force)

    rob_nav._ut_sed(r'networkx\.algorithms\.minors', 'netharn.initializers._nx_ext_v2', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'networkx/networkx/algorithms/minors', 'netharn/netharn/initializers/_nx_ext_v2', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'networkx/algorithms/minors', 'netharn/initializers/_nx_ext_v2', fpath_list=fpath_list, force=force)

    rob_nav._ut_sed(r'networkx\.generators\.random_graphs', 'netharn.initializers._nx_ext_v2.utils', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'networkx\.readwrite\.text', 'netharn.initializers._nx_ext_v2.utils', fpath_list=fpath_list, force=force)

    rob_nav._ut_sed(r'nx.random_tree', 'random_tree', fpath_list=[join(dst, 'utils.py')], force=force)
    rob_nav._ut_sed(r'nx.forest_str', 'forest_str', fpath_list=[join(dst, 'utils.py')], force=force)
    rob_nav._ut_sed(r'nx.random_ordered_tree', 'random_ordered_tree', fpath_list=[join(dst, 'utils.py')], force=force)

    # force = 0
    rob_nav._ut_sed(r'nx.forest_str', 'forest_str', fpath_list=fpath_list, force=force)
    rob_nav._ut_sed(r'nx.random_ordered_tree', 'random_ordered_tree', fpath_list=fpath_list, force=force)

    with open(join(dst, 'tree_embedding.py'), 'a') as file:
        file.write('\n')
        file.write('from netharn.initializers._nx_ext_v2.utils import forest_str  # NOQA\n')
        file.write('from netharn.initializers._nx_ext_v2.utils import random_ordered_tree  # NOQA\n')

    # Enable default autojit
    rob_nav._ut_sed(r'# NETWORKX_AUTOJIT', 'NETWORKX_AUTOJIT', fpath_list=fpath_list, force=force)

    """
    xdoctest ~/code/netharn/netharn/initializers/_nx_ext_v2 all
    """
