
from setuptools import setup

setup(
    name="mddan",
    version="1.0",
    packages=['mddan'], #, 'mddan.dann', 'mddan.abn', 'mddan.adabn', 'mddan.ddcn'],
    package_dir={'mddan': 'mddan'},
    entry_points={
        #'console_scripts': ['app=mddan.app:main', ]
    }
)

