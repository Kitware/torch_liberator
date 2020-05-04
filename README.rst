Torch Liberator - Deploy PyTorch Models 
---------------------------------------

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |Downloads| 

The main webpage for this project is: https://gitlab.kitware.com/computer-vision/torch_liberator

Torch Liberator builds on the "liberator" library to statically extract pytorch
code that defines a model's topology and bundle that with a pretrained weights
file. This results in a single-file deployment package and can potentially
remove dependencies on the codebase used to train the model.

For more info on the base "liberator" package see: https://gitlab.kitware.com/python/liberator

Torch Liberator can also read these deployment files and create an instance of
the model initialized with the correct pretrained weights.


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

