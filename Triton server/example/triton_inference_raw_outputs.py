import requests
import cv2
import numpy as np


def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img


def create_inference_payload(img_data):
    return {
        "inputs": [{
            "name": "data_0", 
            "shape": list(img_data.shape[1:]),
            "datatype": "FP32",
            "data": img_data.flatten().tolist()
        }]
    }


def get_model_prediction(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction request: {e}")
        return None


if __name__ == "__main__":
    MODEL_URL = "http://localhost:8000/v2/models/densenet_onnx/infer"
    
    processed_img = preprocess_image('test.jpg')
    
    payload = create_inference_payload(processed_img)
    
    result = get_model_prediction(MODEL_URL, payload)
    if result:
        print(result)
