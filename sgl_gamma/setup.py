from setuptools import setup, find_packages

setup(
    name='SGL_gamma',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    # other setup configurations
)