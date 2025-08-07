from setuptools import setup, find_packages

setup(
    name="DiffLinker",
    version="1.0",
    packages=find_packages(where="DiffLinker"),  
    package_dir={"": "DiffLinker"}, 
    include_package_data=True,
    install_requires=[
    ]
)