# setup.py

from setuptools import setup, find_packages

# Função para ler as dependências do requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()    
    return [line for line in lines if line and not line.startswith('#')]

setup(
    name='buriedbrains',
    version='0.1.0',
    packages=find_packages(),    
    install_requires=parse_requirements('requirements.txt'),
)