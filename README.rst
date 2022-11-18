Torch Liberator - Deploy PyTorch Models 
---------------------------------------

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |Downloads| 

+----------------------+-------------------------------------------------------------+
| Main Page            | https://gitlab.kitware.com/computer-vision/torch_liberator  |
+----------------------+-------------------------------------------------------------+
| Github Mirror        | https://github.com/Kitware/torch_liberator                  |
+----------------------+-------------------------------------------------------------+
| Pypi                 | https://pypi.org/project/torch_liberator                    |
+----------------------+-------------------------------------------------------------+
| Torch Hackathon 2021 | `Youtube Video`_, `Google Slides`_, and `Submission Page`_  |
+----------------------+-------------------------------------------------------------+

.. _Youtube Video: https://www.youtube.com/watch?v=GQqtn61iNsc
.. _Google Slides: https://docs.google.com/presentation/d/1w9XHkPjtLRj29dw50WP0rSHRRlEfhksP_Sf8XldTSYE
.. _Submission Page: https://devpost.com/software/torchliberator-partial-weight-loading

Torch Liberator is a Python module containing a set of tools for reading and
writing relevant parts of deep networks.

Typically, when moving a deep-network trained with torch, you have to keep
track of the entire codebase that defined the model file in addition to the
checkpoint file containing the learned weights. Torch Liberator contains tools
to extract relevant source code and bundle it with weights and serializing it
into a single-file deployment.

Note: as of torch 1.9, torch comes with a `torch.package <https://pytorch.org/docs/stable/package.html>`__ 
submodule which contains a method for saving model weights with model
structure.  We recommend using ``torch.package`` over the single-file deployments
provided in this package. Thus the ``load_partial_state`` logic is the main
code of interest provided in this module.



Installation
------------

.. code:: bash

    pip install torch_liberator

    # OR with a specific branch

    pip install git+https://gitlab.kitware.com/computer-vision/torch_liberator.git@main


Partial State Loading
---------------------

New in ``0.1.0`` torch liberator now exposes a public ``load_partial_state``
function, which does it best to "shove" weights from one model into another
model. There are several methods to compute associations between layer names in
one model to layer names in another, the most general being the "embedding"
method, and the slightly more structured "isomorphism" option.


Have you ever had the scenario where you use one model as a sub-model in a
bigger network? Then you had to load pretrained subnetwork state into that
bigger model? 

The latest version of ``torch_liberator.load_patial_state`` can handle this by
solving a maximum-common-subtree-isomorphism problem. This computes the largest
possible mapping between the two state dictionaries that share consistent
suffixes.

.. code:: python 

    >>> import torchvision
    >>> import torch
    >>> import torch_liberator
    >>> resnet50 = torchvision.models.resnet50()
    >>> class CustomModel(torch.nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.module = resnet50
    >>>         self.extra = torch.nn.Linear(1, 1)
    >>> # Directly load resnet50 state into a model that has it as an embedded subnetwork
    >>> model = CustomModel()
    >>> model_state_dict = resnet50.state_dict()
    >>> # load partial state returns information about what it did
    >>> info = torch_liberator.load_partial_state(model, model_state_dict, association='isomorphism', verbose=1)
    >>> print(len(info['seen']['full_add']))
    >>> print(len(info['self_unset']))
    >>> print(len(info['other_unused']))
    320
    2
    0
    

It can also handle loading common state between two models that share some
underlying structure.

.. code:: python 

    >>> import torchvision
    >>> import torch
    >>> import torch_liberator
    >>> resnet50 = torchvision.models.resnet50()
    >>> class CustomModel1(torch.nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.module = resnet50
    >>>         self.custom_model1_layer = torch.nn.Linear(1, 1)
    >>> class CustomModel2(torch.nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.backbone = resnet50
    >>>         self.custom_model2_layer = torch.nn.Linear(1, 1)
    >>> # Load as much of model1 state into model2 as possible
    >>> model1 = CustomModel1()
    >>> model2 = CustomModel2()
    >>> model2_state_dict = model2.state_dict()
    >>> # load partial state returns information about what it did
    >>> info = torch_liberator.load_partial_state(model1, model2_state_dict, association='isomorphism', verbose=1)
    >>> print(len(info['seen']['full_add']))
    >>> print(len(info['seen']['skipped']))
    >>> print(len(info['self_unset']))
    >>> print(len(info['other_unused']))
    320
    2
    2
    2


.. code:: python 


    >>> import torchvision
    >>> import torch_liberator
    >>> #
    >>> faster_rcnn = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn()
    >>> resnet50 = torchvision.models.resnet50(pretrained=True)
    >>> state_dict = resnet50.state_dict()
    >>> # Load partial state return a dictionary that tells you how well it did
    >>> info = torch_liberator.load_partial_state(faster_rcnn, state_dict, verbose=0, association='embedding')
    >>> print(ub.map_vals(len, info['seen']))
    >>> print(ub.map_vals(len, ub.dict_diff(info, ['seen'])))
    {'full_add': 265, 'skipped': 55}
    {'other_unused': 55, 'self_unset': 30}

    >>> # Load partial state return a dictionary that tells you how well it did
    >>> info = torch_liberator.load_partial_state(faster_rcnn, state_dict, verbose=0, association='isomorphism')
    >>> print(ub.map_vals(len, info['seen']))
    >>> print(ub.map_vals(len, ub.dict_diff(info, ['seen'])))
    {'full_add': 265, 'skipped': 55}
    {'other_unused': 55, 'self_unset': 30}
    
    

Also, if the sizes of the tensor don't quite fit, they will be mangled, i.e.
"shoved-in" as best as possible. See the docstring for more detail.


Stand-alone Single-File Model Deployments
-----------------------------------------

The original purpose of ``torch_liberator`` was to build standalone torch
packages that contained both the model code and the model weight. It still does
that but ``torch.package`` new in torch 1.9, might be a better solution moving
forward. See `torch.package <https://pytorch.org/docs/stable/package.html>`__
for details.

Torch Liberator builds on the
`liberator <https://gitlab.kitware.com/python/liberator>`__ library to statically
extract pytorch code that defines a model's topology and bundle that with a
pretrained weights file. This results in a single-file deployment package and
can potentially remove dependencies on the codebase used to train the model.

Torch Liberator can also read these deployment files and create an instance of
the model initialized with the correct pretrained weights.

The API is ok, but it does need improvement. However, the current version is in
a working state. There aren't any high level docs, but there are a lot of
docstrings and doctests. The example here gives a good overview of the code by
extracting the AlexNet model from torchvision.


.. code:: python 

    >>> import torch_liberator
    >>> from torch_liberator.deployer import DeployedModel
    >>> from torchvision import models

    >>> print('--- DEFINE A MODEL ---')
    >>> model = models.alexnet(pretrained=False)  # false for test speed
    >>> initkw = dict(num_classes=1000)  # not all models nicely supply this
    >>> model._initkw = initkw
    --- DEFINE A MODEL ---

    >>> print('--- DEPLOY THE MODEL ---')
    >>> zip_fpath = torch_liberator.deploy(model, 'test-deploy.zip')
    --- DEPLOY THE MODEL ---
    [DEPLOYER] Deployed zipfpath=/tmp/tmpeqd3y_rx/test-deploy.zip
    

    >>> print('--- LOAD THE DEPLOYED MODEL ---')
    >>> loader = DeployedModel(zip_fpath)
    >>> model = loader.load_model()
    --- LOAD THE DEPLOYED MODEL ---
    Loading data onto None from <zopen(<_io.BufferedReader name='/tmp/tmpg1kln3kw/test-deploy/deploy_snapshot.pt'> mode=rb)>
    Pretrained weights are a perfect fit
    

The major weirdness right now, is you either have to explicitly define "initkw"
(which are the keyword arguments used to create an instance of our model) at
deploy time, or you can set it as the ``_initkw`` attribute of your model (or
if your keyword arguments all exist as member variables of the class,
torch_liberator tries to be smart and infer what initkw should be).


There is also a torch-liberator CLI that can be used to package a weight file,
a python model file, and optional json metadata.

.. code:: bash

    python -m torch_liberator \
        --model <path-to-the-liberated-py-file> \
        --weights <path-to-the-torch-pth-weight-file> \
        --info <path-to-train-info-json-file> \
        --dst my_custom_deployfile.zip


.. |Pypi| image:: https://img.shields.io/pypi/v/torch_liberator.svg
   :target: https://pypi.python.org/pypi/torch_liberator

.. |Downloads| image:: https://img.shields.io/pypi/dm/torch_liberator.svg
   :target: https://pypistats.org/packages/torch_liberator

.. |ReadTheDocs| image:: https://readthedocs.org/projects/torch_liberator/badge/?version=latest
    :target: http://torch_liberator.readthedocs.io/en/latest/

.. # See: https://ci.appveyor.com/project/jon.crall/torch_liberator/settings/badges
.. .. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/main?svg=true
.. :target: https://ci.appveyor.com/project/jon.crall/torch_liberator/branch/main

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/torch_liberator/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/torch_liberator/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/torch_liberator/badges/main/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/torch_liberator/commits/main

