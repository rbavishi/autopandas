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
    install_requires=[
        "orderedset==2.0.1",
        "Pebble==4.4.0",
        "numpy==1.17.3",
        "astor==0.8.0",
        "tensorflow_gpu==1.9.0",
        "astunparse==1.6.2",
        "pandas==0.23.4",
        "tqdm==4.36.1",
        "python_dateutil==2.8.1",
        "PyYAML==5.3.1",
    ],
)
