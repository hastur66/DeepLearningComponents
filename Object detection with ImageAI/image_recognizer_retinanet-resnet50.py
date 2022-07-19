from imageai.Detection import ObjectDetection

recognizer = ObjectDetection()

path_model = 'models/resnet50_coco_best_v2.1.0.h5'
path_input = 'inputs/sample.jpg'
path_output = 'outputs/detected_image_retina.jpg'

recognizer.setModelTypeAsRetinaNet()

recognizer.setModelPath(path_model)

recognizer.loadModel()

recognition = recognizer.detectObjectsFromImage(
    input_image = path_input,
    output_image_path = path_output
)

for eachItem in recognition:
    print(eachItem['name'], ":", eachItem["percentage_probability"])