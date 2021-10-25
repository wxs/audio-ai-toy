# Timbre Transfer Workshop backend

This repo was built for a brief workshop on creative application of AI as part
of [A-Site workshop](https://asite.krakxr.co/)


## Mac Installation


### 1. Install gsutil

First [install gsutil from here](https://cloud.google.com/storage/docs/gsutil_install).
This will allow you to download data files from Google Cloud.

### Brew other prerequisites

On a Mac install the prerequisites using [https://brew.sh/](Homebrew) and
[Poetry](https://python-poetry.org/). On other other platforms likely
replace homebrew with e.g. `apt`.

N
    brew install libsndfile
    brew install llvm

## Install Python dependenceies
We use [Poetry](https://python-poetry.org/) for Python dependencies. Install
it using pip, and then use it to install other dependencies

    pip install poetry
    poetry update

## Run the demo

Run the following command to start the Flask server
    poetry run flask run

The first time you run this it will download the data files from Google Cloud. Once
complete your server should be running at http://localhost:5000
