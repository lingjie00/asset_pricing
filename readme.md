# Generative Adversarial Network (GAN) Model In Asset Pricing

This repo contains the Python code implementation for my
undergraduate thesis paper. The paper empirically validated
Chen et al. (2021)'s GAN model in the UK LSE 1998-2017 data.

Chen et al. (2021)'s GAN model implemented based on
TensorFlow v1 can be found in their [GitHub
repo](https://github.com/jasonzy121/Deep_Learning_Asset_Pricing)
In this repo we implemented the model using TensorFlow v2.

Chen, L., Pelger, M., & Zhu, J. (2021). Deep learning in
asset pricing. Research Methods & Methodology in Accounting
eJournal.

# Replication environment

We used Python 3.9.0 during development.

Please install all the required libraries with

```bash
$ pip3 install -r requirements.txt
```

# Demonstration

To reproduce the results, please follow the following
procedure:

1. Load all required data in any folder
2. Change the data path in `ap/common.py`
3. Perform ETL with scripts in `etl` in the following order:
    1. `yprice.sync.py`
    2. `fundamental.sync.py`
    3. `factors.sync.py`
    4. `etl.sync.py`
4. Perform training with `training/gan_UK.sync.py`

# Models

Models folder contains all the model implementation.

To change the training settings, there are two files to
change:

1. config.json
2. common.py

`config.json` contains the training config while `common.py`
contains the common variables shared across different
scripts.

We separate the project into the following structure, in the
order to replicate the results

1. ETL
2. Training
3. AP

`ETL` contains the scripts used to transform the raw data.
All transformed data will be saved in a folder named
`data`.

`Training` contains the scripts used to conduct different
trainings presented in the paper. The most relevant script
is `gan_UK.sync.py` which trains the GAN model based on UK
data.

`AP` contains the GAN implementation.
