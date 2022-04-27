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
* packages: streamlit, tensorflow, google_api_python_client, scipy, opencv, scikit-learn

### Installation command
```
conda create --name cc_project python=3.7 
conda activate cc_project 
cd skin_lesion_detection
pip install -r requirements.txt
```

### Deploy Model to Google AI Platform

Similar to README in main branch.

# Usage

In skin_lesion_detection directory:
```
streamlit run app.py
```

# Reference
The code is largely borrowed from the following Github Repos.
Thanks a lot to the authtors for sharing their great work!

[SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL)

[cs329s-ml-deployment-tutorial](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial)

# No Responsibility Disclaimer
The pre-trained model for skin lesion classification and detection in wide field-of-view images are borrowed from [SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL). This repo is only created for demonstrating the feasibility of migrating current research into cloud-computing based applications. The model accuracy is not the main purpose here. Therefore, the prediction result should be carefully used.