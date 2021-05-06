import networkx as nx
from networkx.utils import py_random_state


@py_random_state(1)
def random_tree(n, seed=None, create_using=None):
    """Returns a uniformly random tree on `n` nodes.

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    NetworkX graph
        A tree, given as an undirected graph, whose nodes are numbers in
        the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    Notes
    -----
    The current implementation of this function generates a uniformly
    random Prüfer sequence then converts that to a tree via the
    :func:`~networkx.from_prufer_sequence` function. Since there is a
    bijection between Prüfer sequences of length *n* - 2 and trees on
    *n* nodes, the tree is chosen uniformly at random from the set of
    all trees on *n* nodes.

    Example
    -------
    >>> import networkx as nx
    >>> tree = random_tree(n=10, seed=0)
    >>> print(forest_str(tree, sources=[0]))
    ╙── 0
        ├── 3
        └── 4
            ├── 6
            │   ├── 1
            │   ├── 2
            │   └── 7
            │       └── 8
            │           └── 5
            └── 9

    >>> import networkx as nx
    >>> tree = random_tree(n=10, seed=0, create_using=nx.OrderedDiGraph)
    >>> print(forest_str(tree))
    ╙── 0
        ├─╼ 3
        └─╼ 4
            ├─╼ 6
            │   ├─╼ 1
            │   ├─╼ 2
            │   └─╼ 7
            │       └─╼ 8
            │           └─╼ 5
            └─╼ 9
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph is not a tree")
    # Cannot create a Prüfer sequence unless `n` is at least two.
    if n == 1:
        utree = nx.empty_graph(1)
    else:
        sequence = [seed.choice(range(n)) for i in range(n - 2)]
        utree = nx.from_prufer_sequence(sequence)

    if create_using is None:
        tree = utree
    else:
        # TODO: maybe a tree classmethod like
        # Graph.new, Graph.fresh, or something like that
        def new(cls_or_self):
            if hasattr(cls_or_self, "_adj"):
                # create_using is a NetworkX style Graph
                cls_or_self.clear()
                self = cls_or_self
            else:
                # try create_using as constructor
                self = cls_or_self()
            return self

        tree = new(create_using)
        if tree.is_directed():
            # Use a arbitrary root node and dfs to define edge directions
            edges = nx.dfs_edges(utree, source=0)
        else:
            edges = utree.edges

        # Populate the specified graph type
        tree.add_nodes_from(utree.nodes)
        tree.add_edges_from(edges)

    return tree


@py_random_state(2)
def random_ordered_tree(n, seed=None, directed=False):
    """
    Creates a random ordered tree

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    directed : bool
        if the edges are one-way

    Returns
    -------
    networkx.OrderedDiGraph | networkx.OrderedGraph

    Example
    -------
    >>> import networkx as nx
    >>> assert len(random_ordered_tree(n=1, seed=0).nodes) == 1
    >>> assert len(random_ordered_tree(n=2, seed=0).nodes) == 2
    >>> assert len(random_ordered_tree(n=3, seed=0).nodes) == 3
    >>> otree = random_ordered_tree(n=5, seed=3, directed=True)
    >>> print(forest_str(otree))
    ╙── 0
        └─╼ 1
            └─╼ 4
                ├─╼ 2
                └─╼ 3
    """
    from networkx.utils import create_py_random_state

    rng = create_py_random_state(seed)
    # Create a random undirected tree
    create_using = nx.OrderedDiGraph if directed else nx.OrderedGraph
    otree = random_tree(n, seed=rng, create_using=create_using)
    return otree


def forest_str(graph, with_labels=True, sources=None, write=None):
    """
    Creates a nice utf8 representation of a directed forest

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    with_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Example
    -------
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(forest_str(graph))
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   │   ├─╼ 7
        │   │   └─╼ 8
        │   └─╼ 4
        │       ├─╼ 9
        │       └─╼ 10
        └─╼ 2
            ├─╼ 5
            │   ├─╼ 11
            │   └─╼ 12
            └─╼ 6
                ├─╼ 13
                └─╼ 14


    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(forest_str(graph))
    ╙── 0
        └── 1
            └── 2
    """
    import networkx as nx

    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    if len(graph.nodes) == 0:
        _write("╙")
    else:
        if not nx.is_forest(graph):
            raise nx.NetworkXNotImplemented("input must be a forest or the empty graph")

        is_directed = graph.is_directed()
        succ = graph.succ if is_directed else graph.adj

        if sources is None:
            if is_directed:
                # use real source nodes for directed trees
                sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            else:
                # use arbitrary sources for undirected trees
                sources = [
                    min(cc, key=lambda n: graph.degree[n])
                    for cc in nx.connected_components(graph)
                ]

        # Populate the stack with each source node, empty indentation, and mark
        # the final node. Reverse the stack so sources are popped in the
        # correct order.
        last_idx = len(sources) - 1
        stack = [(node, "", (idx == last_idx)) for idx, node in enumerate(sources)][
            ::-1
        ]

        seen = set()
        while stack:
            node, indent, islast = stack.pop()
            if node in seen:
                continue
            seen.add(node)

            # Notes on available box and arrow characters
            # https://en.wikipedia.org/wiki/Box-drawing_character
            # https://stackoverflow.com/questions/2701192/triangle-arrow
            if not indent:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if islast:
                    this_prefix = indent + "╙── "
                    next_prefix = indent + "    "
                else:
                    this_prefix = indent + "╟── "
                    next_prefix = indent + "╎   "

            else:
                # For individual forests distinguish between directed and
                # undirected cases
                if is_directed:
                    if islast:
                        this_prefix = indent + "└─╼ "
                        next_prefix = indent + "    "
                    else:
                        this_prefix = indent + "├─╼ "
                        next_prefix = indent + "│   "
                else:
                    if islast:
                        this_prefix = indent + "└── "
                        next_prefix = indent + "    "
                    else:
                        this_prefix = indent + "├── "
                        next_prefix = indent + "│   "

            if with_labels:
                label = graph.nodes[node].get("label", node)
            else:
                label = node

            _write(this_prefix + str(label))

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            children = [child for child in succ[node] if child not in seen]
            for idx, child in enumerate(children[::-1], start=1):
                islast_next = idx <= 1
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)
