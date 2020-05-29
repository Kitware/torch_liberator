Torch Liberator - Deploy PyTorch Models 
---------------------------------------

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |Downloads| 

+------------------+------------------------------------------------------------+
| Main Page        | https://gitlab.kitware.com/computer-vision/torch_liberator |
+------------------+------------------------------------------------------------+
| Github Mirror    | https://github.com/Kitware/torch_liberator                 |
+------------------+------------------------------------------------------------+
| Pypi             | https://pypi.org/project/torch_liberator                   |
+------------------+------------------------------------------------------------+


Torch Liberator builds on the "liberator" library to statically extract pytorch
code that defines a model's topology and bundle that with a pretrained weights
file. This results in a single-file deployment package and can potentially
remove dependencies on the codebase used to train the model.

For more info on the base "liberator" package see: https://gitlab.kitware.com/python/liberator or https://github.com/Kitware/liberator

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


.. |Pypi| image:: https://img.shields.io/pypi/v/torch_liberator.svg
   :target: https://pypi.python.org/pypi/torch_liberator

.. |Downloads| image:: https://img.shields.io/pypi/dm/torch_liberator.svg
   :target: https://pypistats.org/packages/torch_liberator

.. |ReadTheDocs| image:: https://readthedocs.org/projects/torch_liberator/badge/?version=latest
    :target: http://torch_liberator.readthedocs.io/en/latest/

.. # See: https://ci.appveyor.com/project/jon.crall/torch_liberator/settings/badges
.. .. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
.. :target: https://ci.appveyor.com/project/jon.crall/torch_liberator/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/torch_liberator/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/torch_liberator/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/torch_liberator/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/torch_liberator/commits/master

