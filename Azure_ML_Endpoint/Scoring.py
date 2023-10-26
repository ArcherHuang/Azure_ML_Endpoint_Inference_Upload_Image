import os
import cv2
import json
import requests
from PIL import Image
import tensorflow as tf
from datetime import datetime
from azure.storage.blob import BlobClient

from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def load_label(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}
        
def predict(image_path, name, file_extension):
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # load label
    label_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Models/coco_labels.txt")
    labels = load_label(label_path)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 300, 300, 3])

    # Get the width and height of the image
    image_height, image_width, channels = img.shape

    # set frame as input tensors
    model.set_tensor(input_details[0]['index'], img_rgb)

    # perform inference
    model.invoke()

    # Get output tensor
    boxes = model.get_tensor(output_details[0]['index'])[0]
    classes = model.get_tensor(output_details[1]['index'])[0]
    scores = model.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * image_height)))
            xmin = int(max(1, (boxes[i][1] * image_width)))
            ymax = int(min(image_height, (boxes[i][2] * image_height)))
            xmax = int(min(image_width, (boxes[i][3] * image_width)))

            cv2.rectangle(img, (xmin, ymin),
                                (xmax, ymax), (10, 255, 0), 4)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            print(f"label: {label}")
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (
                        xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Save the image
            file_name = f"{name}-inference{file_extension}"
            cv2.imwrite(file_name, img)

    try:
        BLOB_CONNECTION_STRING = ""
        BLOB_CONTAINER_NAME = "images"
        blob = BlobClient.from_connection_string(
            conn_str = BLOB_CONNECTION_STRING,
            container_name = BLOB_CONTAINER_NAME,
            blob_name = file_name
        )
        with open(file_name, "rb") as data:
            blob.upload_blob(data, overwrite=True)
        return f"file_name: {file_name}, label: {label}"

    except Exception as ex:
        print(f"Exception: {ex}")

def init():
    print(f"AZUREML_MODEL_DIR: {os.getenv('AZUREML_MODEL_DIR')}")
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Models/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    model = tf.lite.Interpreter(model_path=model_path)
    
@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)

    try:
        byte_data = request.get_data(False)
        json_object = json.loads(str(byte_data, 'utf-8'))
        file_name = json_object['fileUrl'].split("/")[-1]
        name, file_extension = os.path.splitext(file_name)
        response = requests.get(json_object['fileUrl'])
        if response.status_code == 200:
            new_file_name = f"{name}-raw{file_extension}"
            with open(new_file_name, 'wb') as file:
                file.write(response.content)
            print(f'File saved as {new_file_name}')
            preds = predict(new_file_name, name, file_extension)
            return AMLResponse(json.dumps(preds), 200)
        else:
            print(f'Failed to download the file. HTTP status code: {response.status_code}')
            return AMLResponse(f"Failed to download the file. HTTP status code: {response.status_code}", 500)
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return AMLResponse(f"Error: {str(e)}", 500)
