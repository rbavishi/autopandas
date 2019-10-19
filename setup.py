import glob

from setuptools import setup, find_packages

setup(
    name='autopandas_v2',
    description='Synthesis for Pandas',
    author='Rohan Bavishi, Caroline Lemieux',
    author_email='rbavishi@berkeley.edu, clemieux@berkeley.edu',
    version='1.0.0',
    packages=find_packages(),
    package_data={'': ['*.py']},
    data_files=[('config', glob.glob('config/*', recursive=True))],
    include_package_data=True,
    entry_points={'console_scripts': ['autopandas_v2=autopandas_v2.main:run_console']},
    zip_safe=False,
    install_requires=['pebble', 'astor', 'astunparse', 'orderedset', 'pandas==0.23.4', 'tqdm', 'PyYAML']
)
