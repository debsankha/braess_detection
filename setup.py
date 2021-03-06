from setuptools import setup, find_packages

setup(
    name="braess_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "networkx==2.0",
        "numpy",
        "pandas",
        "seaborn",
        "flownetpy",
        "pytest",
        "jupyterlab",
        "jupyterlab_vim",
        "tqdm",
        "loky",
    ],
)
