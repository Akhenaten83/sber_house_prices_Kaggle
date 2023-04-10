# sber_house_prices_Kaggle
complete data analysis and model
## Installation
```bash
pip install -e .
```
## Repository Structure


```
├── README.md
│
├── ml_mfp                     <- ml-mfp package.  
│   │
│   ├── docker                 <- API to access train.py and predict.py modules of the ml-mfp.model package through
│   │                             the ml-mfp docker container providing an environment with all the requred ml-mfp.model
│   │                             package dependencies installed.  
│   │
│   ├── model                  <- ml-mfp.model package: model construction, feature generation, and train/predict modules;
│   │                             The modules won't work until "ml-mfp[model]" dependency set is installed.
│   │
│   ├── plotting               <- Scripts for visualization;
│   │                             For proper functioning, "ml-mfp[plotting]" dependency set must be installed.
│   └── data                   <- extra source distribution files.
│
├── requirements               <- requirements files.
│   │  
│   ├── ml_mfp.base.txt        <- common for all ml-mfp subpackages.
│   ├── ml_mfp.txt             <- core ml-mfp dependecy set.
│   ├── ml_mfp.model.txt       <- ml-mfp[model] dependecy set.
│   ├── ml_mfp.plotting.txt    <- ml-mfp[plotting] dependecy set.
│   ├── streamlit_app.txt      <- streamlit apllication located in streamlit_app/.
│   ├── telegram.txt           <- requirements for scripts located in cli/telegram/.
│   └── docs.txt               <- requirements for generating Sphinx documentation.
│
├── cli                        <- command-line interface for training models, updating MFP bags, detecting updated MFP bags,
│                                 and reporting to the MFP telegram channel.
│
├── Dockerfile                 <- instructions for ml-mfp docker image construction [docker-rnd.bostongene.internal/ml-mfp].
│
├── Makefile                   <- instructions for ml-mfp docker image build and push to the registry.
│
├── jenkins                    <- Jenkins Pipeline executables: tests and package release to nexus.
│
├── Jenkinsfile                <- Definition of Jenkins Pipeline.
│
└── upload_repo.sh             <- Manual upload of package to Nexus registry.
```
------------