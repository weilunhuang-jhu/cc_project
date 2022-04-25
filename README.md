# Introduction

Teledermatology is in an unmet need as skin cancer is one of the most common cancer diseases worldwide. Cloud-computing is key to enable a global solution with current advancement in deep-learning based skin lesion detection/classification systems. With recent advancement in deep-learning based medical AI, we want to enable some of the current research into the cloud.

This project provides an implementation of deploying deep-learning based skin lesion detection of clinical images into the cloud (GCP). An example was implemented with the research from [Soenksen et. al, Using deep learning for dermatologist-level detection of suspicious pigmented skin lesions from wide-field images, Science Translational Medicine 2021](https://pubmed.ncbi.nlm.nih.gov/33597262/). This project also has experimented with the performance and scalability for the models deployed on GCP AI Platform.

Currently support:

* On Google APP Engine:

End-to-end skin lesion classifiction into 6 classes: Background, skin-edge, skin, NSPL-A (non-suspicious skin lesion), NSPL-B (non-suspicious skin lesion for follow-up), and SPL (suspicious skin lesion). 


* In local APP:
End-to-end skin lesion classification model and ugly duckling detection.

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

### Deploy Model to GCP AI Platform
To be continued

### Deploy to GCP APP Engine
To be continued

# Usage

In skin_lesion_detection directory:
In "lumo_clinician_app" folder:
```
streamlit run app.py
```

# Reference
The code is largely borrowed from the following Github Repos.
Thanks a lot for the authtors for sharing their great work!

[SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL)

[cs329s-ml-deployment-tutorial](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial)

# No Responsibility Disclaimer
The pre-trained model for skin lesion classification and detection in wide field-of-view images are borrowed from [SPL_UD_DL](https://github.com/lrsoenksen/SPL_UD_DL). This repo is only created for demonstrating the feasibility of migrating current research into cloud-computing based applications. The model accuracy is not the main purpose here. Therefore, the prediction result should be carefully used.