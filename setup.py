#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os.path import exists

from setuptools import find_packages

# if exists('CMakeLists.txt'):
try:
    import os
    val = os.environ.get('DISABLE_C_EXTENSIONS', '').lower()
    flag = val in {'true', 'on', 'yes', '1'}

    if '--universal' in sys.argv:
        flag = True

    if '--disable-c-extensions' in sys.argv:
        sys.argv.remove('--disable-c-extensions')
        flag = True

    if flag:
        # Hack to disable all compiled extensions
        from setuptools import setup
    else:
        from skbuild import setup
except ImportError:
    setup = None


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class Finished(Exception):
        pass
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
                    raise Finished
    visitor = VersionVisitor()
    try:
        visitor.visit(pt)
    except Finished:
        pass
    return visitor.version


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''


def parse_requirements(fname='requirements.txt', versions=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        versions (bool | str, default=False):
            If true include version specs.
            If strict, then pin to the minimum version.

    Returns:
        List[str]: list of requirements items
    """
    from os.path import exists, dirname, join
    import re
    require_fpath = fname

    def parse_line(line, dpath=''):
        """
        Parse information from a line in a requirements text file

        line = 'git+https://a.com/somedep@sometag#egg=SomeDep'
        line = '-e git+https://a.com/somedep@sometag#egg=SomeDep'
        """
        # Remove inline comments
        comment_pos = line.find(' #')
        if comment_pos > -1:
            line = line[:comment_pos]

        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = join(dpath, line.split(' ')[1])
            for info in parse_require_file(target):
                yield info
        else:
            # See: https://www.python.org/dev/peps/pep-0508/
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                if ';' in line:
                    pkgpart, platpart = line.split(';')
                    # Handle platform specific dependencies
                    # setuptools.readthedocs.io/en/latest/setuptools.html
                    # #declaring-platform-specific-dependencies
                    plat_deps = platpart.strip()
                    info['platform_deps'] = plat_deps
                else:
                    pkgpart = line
                    platpart = None

                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, pkgpart, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        dpath = dirname(fpath)
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line, dpath=dpath):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if versions and 'version' in info:
                    if versions == 'strict':
                        # In strict mode, we pin to the minimum version
                        if info['version']:
                            # Only replace the first >= instance
                            verstr = ''.join(info['version']).replace('>=', '==', 1)
                            parts.append(verstr)
                    else:
                        parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    plat_deps = info.get('platform_deps')
                    if plat_deps is not None:
                        parts.append(';' + plat_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def clean():
    import ubelt as ub
    import os
    import glob
    from os.path import dirname, join

    modname = 'torch_liberator'
    repodir = dirname(os.path.realpath(__file__))

    toremove = []
    for root, dnames, fnames in os.walk(repodir):

        if os.path.basename(root) == modname + '.egg-info':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '__pycache__':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '_ext':
            # Remove torch extensions
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == 'build':
            # Remove python c extensions
            if len(dnames) == 1 and dnames[0].startswith('temp.'):
                toremove.append(root)
                del dnames[:]

        # Remove simple pyx inplace extensions
        for fname in fnames:
            if fname.endswith('.pyc'):
                toremove.append(join(root, fname))
            if fname.endswith(('.so', '.c', '.o')):
                if fname.split('.')[0] + '.pyx' in fnames:
                    toremove.append(join(root, fname))

    def enqueue(d):
        if exists(d) and d not in toremove:
            toremove.append(d)

    for d in glob.glob(join(repodir, 'torch_liberator/_nx_ext_v2/*.pkl')):
        enqueue(d)

    for d in glob.glob(join(repodir, 'torch_liberator/_nx_ext_v2/*.pkl.meta')):
        enqueue(d)

    for d in glob.glob(join(repodir, 'torch_liberator/_nx_ext_v2/*.cpp')):
        enqueue(d)

    for d in glob.glob(join(repodir, 'torch_liberator/_nx_ext_v2/*.so')):
        enqueue(d)

    enqueue(join(repodir, 'torch_liberator.egg-info'))
    enqueue(join(repodir, 'pip-wheel-metadata'))

    for dpath in toremove:
        ub.delete(dpath, verbose=1)


NAME = 'torch_liberator'
VERSION = parse_version('torch_liberator/__init__.py')


if __name__ == '__main__':
    if 'clean' in sys.argv:
        clean()

    setupkw = {}
    setupkw["install_requires"] = parse_requirements("requirements/runtime.txt")
    setupkw["extras_require"] = {
        "all": parse_requirements("requirements.txt"),
        "tests": parse_requirements("requirements/tests.txt"),
        "optional": parse_requirements("requirements/optional.txt"),
        "all-strict": parse_requirements("requirements.txt", versions="strict"),
        "runtime-strict": parse_requirements(
            "requirements/runtime.txt", versions="strict"
        ),
        "tests-strict": parse_requirements("requirements/tests.txt", versions="strict"),
        "optional-strict": parse_requirements(
            "requirements/optional.txt", versions="strict"
        ),
    }

    setup(
        name=NAME,
        version=VERSION,
        author='joncrall',
        author_email='jon.crall@kitware.com',
        description=('The torch_liberator Module'),
        url='https://gitlab.kitware.com/computer-vision/torch_liberator',
        long_description=parse_description(),
        long_description_content_type='text/x-rst',
        license='Apache 2',
        packages=find_packages('.'),
        python_requires=">=3.6",
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Utilities',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        **setupkw,
    )
