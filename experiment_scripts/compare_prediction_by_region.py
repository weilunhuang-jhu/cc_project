# Script to get prediction time of Google AI Platform (CPU) in different regions

import os
import sys
import glob
import time
import re
import numpy as np
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

def get_image(path):
    """ Read image to np array and then turn to list object. (for json) """
    img = image.load_img(path, target_size=(150, 150, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x.tolist()
    return x

def predict_json_online(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)
    # print(model_path)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=True, client_options=client_options).projects() # cache_discovery=False
    instances_list = instances # turn input into list (ML Engine wants JSON)
    input_data_json = {"instances": instances_list} 

    # input_data_json = {"instances" : []}
    # for i,instance in enumerate(instances_list):
    #     input_data_json["instances"].append(instance)

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

def main():

    models_global = ["vggg_16_asia", "vggg_16_europe", "vggg_16_east_4", "vggg_16_central_1"]
    models_regional = ["vggg_16_asia_northeast1_regional", "vggg_16_europe_west_1_regional", "vggg_16_east_4_regional", "vggg_16_central_1_regional"]

    # Setup environment credentials (you'll need to change these)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ccproject-343606-498535894d22.json" # change for your GCP key
    PROJECT = "ccproject-343606" # change for your GCP project
    REGIONs = ["asia-northeast1", "europe-west1", "us-east4", "us-central1"]

    # load image
    Blob_folder = "../data/Blobs"
    fnames = glob.glob(Blob_folder + "/*.png")
    fnames.sort()
    fname = fnames[0]
    img = get_image(fname)
    num_img = 10 # change number of image here for experiment

    # Get Prediction time
    print("Using global end-points")
    for model in models_global:
        t0 = time.perf_counter()
        for i in range(num_img):
            preds = predict_json_online(project=PROJECT,
                                    region=None,
                                    model=model,
                                    instances=img)
        t1 = time.perf_counter()
        print(model + " : " + str(t1-t0))

    print("Using regional end-points")
    for model,region in zip(models_regional, REGIONs):
        t0 = time.perf_counter()
        for i in range(num_img):
            preds = predict_json_online(project=PROJECT,
                                    region=region,
                                    model=model,
                                    instances=img)
        t1 = time.perf_counter()
        print(model + " : " + str(t1-t0))

if __name__ == "__main__":
    main()
