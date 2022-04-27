# Introduction

**Coures Project of Cloud Computing EN.601.619 Spring 2022**

Teledermatology is in an unmet need as skin cancer is one of the most common cancer diseases worldwide. Cloud-computing is key to enable a global solution with current advancement in deep-learning based skin lesion detection/classification systems. This project provides an implementation of deploying deep-learning based skin lesion detection of clinical images into the cloud (ex: Google Cloud Platform (GCP) here). An example was implemented with the research from [Soenksen et. al, Using deep learning for dermatologist-level detection of suspicious pigmented skin lesions from wide-field images, Science Translational Medicine 2021](https://pubmed.ncbi.nlm.nih.gov/33597262/). This project also has experimented with the performance and scalability for the models deployed on Google AI Platform.

**Currently supported:**

* On Google APP Engine:
> End-to-end skin lesion classifiction into 6 classes: Background, skin-edge, skin, NSPL-A (non-suspicious skin lesion), NSPL-B (non-suspicious skin lesion to follow), and SPL (suspicious skin lesion). 


* On local APP: (ugly_duckling branch)
> End-to-end skin lesion classification model and ugly duckling detection (highlight suspicous skin lesions based on ugly duckling criteria in wide field-of-view images)

# Installation:

### Dependency

* python: 3.7
* packages: streamlit, tensorflow, google_api_python_client

### Installation command
```
conda create --name cc_project python=3.7 
conda activate cc_project 
cd skin_lesion_detection
pip install -r requirements.txt
```

### Deploy Model to Google AI Platform

It requires several steps below to deploy the model on Google AI Platform:
> * A Google Storage bucket to store the trained model
> * A hosted model on Google AI Platform (create model, manage versions)
> * A service key to access the hosted model on Google AI Platform

More detailed information can be found in [this wonderful tutorial](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial)

Models can be found in the [SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL) repo, the _finetuned vgg16_ model is used in this project.

### Deploy Application to Google APP Engine

It requires several steps below to deploy the application on Google APP Engine:
> * Put APP into a Docker container (becomes a Docker image)
> * Upload Docker image to Google Container Registry to be deployed as an APP Engine Instance

Inside the skin_lesion_detection directory,
```
make gcloud-deploy
```

More detailed information can be found in [this wonderful tutorial](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial)

# Usage

In skin_lesion_detection directory:
```
streamlit run app.py
```

# Experiment:

Experiment of the latency and scalability performance of online inference of models deployed on Google AI Platform.

* Latency experiment by comparing the performance between different machine types (cpu/gpu) and different model endpoint regions: *experiment_scripts/compare_prediction_by_machine_types.py* and *experiment_scripts/compare_prediction_by_region.py*
> Deploy the model in different machine types and endpoint regions, modify the model names in the scripts, run the python scripts.

* Scalability experiment by comparing the performance between different number of simultaneous requests (processes) and using different number of minimum nodes: *experiment_scripts/saclability_measuremet_multiprocess.py* and *experiment_scripts/scalability_measurement_different_node_num.py*
> Deploy the model with different number of minimum nodes hosting the model service, modify the model names in the scripts, run the python scripts. This project compares the scalability performance among single machine/two machines under the same networks/two machines under different networks.


# Reference
The code is largely borrowed from the following Github Repos.
Thanks a lot to the authtors for sharing their great work!

[SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL)

[cs329s-ml-deployment-tutorial](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial)

# No Responsibility Disclaimer
The pre-trained model for skin lesion classification and detection in wide field-of-view images are borrowed from [SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL). This repo is only created for demonstrating the feasibility of migrating current research into cloud-computing based applications. The model accuracy is not the main purpose here. Therefore, the prediction result should be carefully used.