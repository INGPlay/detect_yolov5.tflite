try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import cv2
import os
import sys
import timeit
from detect.detector import detectImage


def main() :

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    imageSize = 320
    thres = 0.5

    url = "People - 84973.mp4"
    model = 'yolov5n-int8-320_edgetpu.tflite'

    cap = cv2.VideoCapture(url)

    videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Width : {videoWidth}, Height : {videoHeight}, evaluedFps : {fps}")

    # try load_delegate('libedgetpu.so.1')
    try:
        delegate = load_delegate('libedgetpu.so.1')
    except ValueError:
        delegate = 0
    
    if delegate :
        interpreter = Interpreter(
            model_path=model,
            experimental_delegates=[delegate])
    else :
        interpreter = Interpreter(model_path=model)

    interpreter.allocate_tensors()

    inputDetails = interpreter.get_input_details()[0]
    outputDetails = interpreter.get_output_details()[0]
    if not cap.read() :
        print("Can not read")
        sys.exit()

    # Label load
    labels = []
    with open("labels.txt", "r") as fs :
        lines = fs.readlines()
        for l in lines :
            l = l.strip()
            labels.append(l)

    frameSecond = 1.0/fps
    endTime = 0
    # Frame repeat
    while True :
        # timer start
        frameStartTime = timeit.default_timer()

        ret, image = cap.read()
        if not ret :
            break 
        
        # Don't think about zero frame and fps over
        if frameSecond > frameStartTime - endTime :
            continue

        outputImage = detectImage(
            image= image,
            interpreter= interpreter,
            inputDetails= inputDetails,
            outputDetails= outputDetails,
            imageSize= imageSize,
            thres= thres,
            labels= labels
        )
        
        cv2.imshow("detect", outputImage)

        # timer end
        endTime = timeit.default_timer()
        evaluedFps = int(1./(endTime - frameStartTime))
        print(f"evaluedFps : {evaluedFps}")

        if cv2.waitKey(1) == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()



main()
