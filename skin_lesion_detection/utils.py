# Utils for preprocessing data etc 
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

lesion_classes = [
    'SPL',
    'NSPL-B',
    "NSPL-A",
    "Skin",
    "Skin edge",
    "Background"]

classes_and_models = {
    "model_1": {
        "classes": lesion_classes,
        "model_name": "cc_project_skin_lesion_vgg16" # change to be your model name
    },
    "model_2": {
        "classes": lesion_classes,
        "model_name": "cc_project_ugly_duckling"
    }
}

def predict_json(project, region, model, instances, version=None):
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

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    x = instances.numpy()
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    instances_list = x.tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model
def get_image_tf(filename):
    img = tf.io.decode_image(filename, channels=3) # make sure there's 3 colour channels (for PNG's)
    img = tf.image.resize(img, [150, 150])
    # x = image.img_to_array(img)
    return img

# Create a function to import an image to np array
def get_image_np(filename):
    img = tf.io.decode_image(filename, channels=3) # make sure there's 3 colour channels (for PNG's)
    img = img.numpy().transpose(1,0,2) # make to HWC align with opencv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print("debug")
    # print(img.shape)
    # cv2.imwrite("test.png", img)
    return img

def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger
