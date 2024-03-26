from setuptools import setup, find_packages

setup(
    name='Synthetic-Codes',
    version='0.0.1',
    install_requires = [
        'boto3==1.26.63',
        'pydantic==1.10.2',
        "typer==0.9.0" 
    ],
    packages=find_packages()
)