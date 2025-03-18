from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="keras_leaderboard",
    version="0.1.0",
    author="Catherine Ordun",
    author_email="",  # Add your email here
    description="A framework for tracking and comparing Keras model performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nudro/keras_leaderboard",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'keras-leaderboard=keras_leaderboard.cli:main',
        ],
    },
) 