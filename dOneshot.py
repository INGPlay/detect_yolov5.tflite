import detect.detector as detector
try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image4.png')
model = 'yolov5n-int8-320_edgetpu.tflite'
interpreter = Interpreter(
    model_path=model,
    experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()[0]
outputDetails = interpreter.get_output_details()[0]
imageSize = 320
thres = 0.5
labels = []
with open("labels.txt", "r") as fs :
    lines = fs.readlines()
    for l in lines :
        l = l.strip()
        labels.append(l)

setImage = detector.detectImage(
    image=image,
    interpreter=interpreter,
    inputDetails=inputDetails,
    outputDetails=outputDetails,
    imageSize=imageSize,
    thres=thres,
    labels=labels
)

plt.imshow(setImage); plt.show()