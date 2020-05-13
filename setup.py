from setuptools import setup, find_packages

print(find_packages())
setup(name='adversarial_explanations', version='0.1', packages=find_packages(), install_requires=['numpy',
                                                                                                  'matplotlib',
                                                                                                  'sklearn'])
