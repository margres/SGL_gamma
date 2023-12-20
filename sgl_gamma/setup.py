from setuptools import setup

setup(
    name='sgl_gamma',
    version='0.1',
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    # other setup configurations
)