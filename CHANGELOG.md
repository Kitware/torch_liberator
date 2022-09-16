# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.2.0 - Unreleased

### Changed:

* Moved implementation of the backend maximum common subtree algorithm to `https://github.com/Erotemic/networkx_algo_common_subtree`.
* The weight mapping is now returned by the partial state loading and pretrained forward
* Changed defaults to use the isomorphism algorithm by default and other tweaks.


## Version 0.1.1 - Released 2021-11-01

### Changed

* Using scikit-build and CMake to build cython modules and publishing wheels
  instead of hacky autojit solutions.


## Version 0.1.0 - Released 2021-05-06

### Added
* Ported the maximum common subtree isomorphism building associations in `load_partial_state`.


## Version 0.0.5 - Released 2021-05-06

### Added
* Added `cli` for manually making deployed zip files

### Changed
* Removed Python 2.7 and 3.5 Support.
* The first part in the deployed name is now based on the netharn run "name"
  rather than the name of the model class

### Fixed
* Support new netharn checkpoints dir


## Version 0.0.4 - Released 2021-05-06

### Added
* Added `extract_snapshot` method to `DeployedModel`


## Version 0.0.3 - Released


## [Version 0.0.1] - 

### Added
* Initial version

