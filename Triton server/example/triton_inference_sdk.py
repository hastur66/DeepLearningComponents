import tritonclient.http as tritonhttpclient
import cv2
import numpy as np
import os


def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img


def create_inference_input(img_data):
    inputs = []
    inputs.append(tritonhttpclient.InferInput('data_0', list(img_data.shape), "FP32"))
    inputs[0].set_data_from_numpy(img_data)
    return inputs


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def get_model_prediction(client, inputs):
    try:
        result = client.infer("densenet_onnx", inputs)
        return result
    except Exception as e:
        print(f"Error making prediction request: {e}")
        return None


def parse_predictions(result, labels, top_k=5):
    predictions = result.as_numpy('output_0')
    predictions = np.squeeze(predictions)
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    return [(labels[i], predictions[i]) for i in top_indices]


if __name__ == "__main__":
    TRITON_URL = "localhost:8000"
    LABELS_PATH = "../models/densenet_onnx/densenet_labels.txt"
    
    # Create triton client
    client = tritonhttpclient.InferenceServerClient(url=TRITON_URL)
    
    labels = load_labels(LABELS_PATH)
    processed_img = preprocess_image('test.jpg')
    inputs = create_inference_input(processed_img)
    
    result = get_model_prediction(client, inputs)
    if result:
        predictions = parse_predictions(result, labels)
        for class_name, confidence in predictions:
            print(f"{class_name}: {confidence:.4f}")
