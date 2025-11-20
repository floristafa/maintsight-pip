#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

# Read version from __init__.py
def read_version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

setup(
    name='maintsight',
    version=read_version(),
    description='AI-powered maintenance risk predictor for git repositories using XGBoost',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='TechDebtGPT Team',
    author_email='support@techdebtgpt.com',
    url='https://github.com/techdebtgpt/maintsight',
    project_urls={
        'Bug Tracker': 'https://github.com/techdebtgpt/maintsight/issues',
        'Documentation': 'https://github.com/techdebtgpt/maintsight#readme',
        'Source Code': 'https://github.com/techdebtgpt/maintsight',
    },
    packages=find_packages(exclude=['tests*']),
    package_data={
        'models': ['*.pkl', '*.json'],
        'utils': ['templates/*.html'],
    },
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'click>=8.0.0',
        'dill>=0.3.6',
        'gitpython>=3.1.0',
        'joblib>=1.3.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.1.0',
        'xgboost>=1.6.0',
        'jinja2>=3.0.0',
        'rich>=12.0.0',
        'tqdm>=4.62.0',
        'typing-extensions>=4.0.0; python_version<"3.10"',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'pre-commit>=2.15.0',
        ],
        'html': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'maintsight=cli:main',
            'ms=cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Version Control :: Git',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='maintenance technical-debt risk-prediction xgboost git code-quality machine-learning repository-health',
    license='Apache-2.0',
    zip_safe=False,
)