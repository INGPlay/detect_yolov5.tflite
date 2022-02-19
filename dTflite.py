try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import numpy as np
import cv2
import os
import sys
import timeit
import platform


def cropImageCenter(image, width, height) :
    imageShape = image.shape
    originalWidth = int(imageShape[0])
    originalHeight = int(imageShape[1])
    sliX = int((originalWidth - width) * 0.5)
    sliY = int((originalHeight - height) * 0.5)

    image = image[sliX : sliX + width, sliY : sliY + height]
    
    return image

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
          
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    centerX, centerY = pos
    centerY = centerY - text_h
    cv2.rectangle(img, (centerX,centerY), (centerX + text_w, centerY + text_h), text_color_bg, -1)
    cv2.putText(img, text, (centerX, centerY + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def detectImage(image, interpreter, inputDetails, imageSize, thres, labels) :

    image = cropImageCenter(image=image, width=imageSize, height=imageSize)
    # plt.imshow(image); plt.show()

    # quantinzation
    scale, zeroPoint = inputDetails["quantization"]

    qImage = np.uint8(image / scale + zeroPoint)

    #input
    interpreter.set_tensor(inputDetails['index'], np.uint8([qImage]))

    interpreter.invoke()

    # output
    outputDetails = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(outputDetails['index'])[0]
    scale, zeroPoint = inputDetails["quantization"]
    output = (output.astype(np.float32) - zeroPoint) * scale

    for i in range(len(output)) :
        conf = output[i][4]

        if conf >= thres :
            w = round(output[i][2] * imageSize)
            h = round(output[i][3] * imageSize)
            x = round(output[i][0] * imageSize)
            y = round(output[i][1] * imageSize)

            cv2.rectangle(image, (int(x),int(y),int(w),int(h)),(0,255,0), 2)
            cls = output[i][5:].argmax()
            label = labels[cls]

            draw_text(image, f"{label} {conf:.2f}",
                pos= (int(x), int(y)), 
                font= cv2.FONT_HERSHEY_DUPLEX, 
                font_scale = 1, 
                text_color = (0, 0, 255), 
                font_thickness = 1
            )

    cv2.imshow("video", image)


def main() :

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    imageSize = 320
    thres = 0.5

    #url = "rtsp://admin:worldcns\!@192.168.1.12:554/profile1/media.smp"
    url = "People - 84973.mp4"
    model = 'yolov5n-int8-320_edgetpu.tflite'        

    cap = cv2.VideoCapture(url)

    videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Width : {videoWidth}, Height : {videoHeight}, evaluedFps : {fps}")

    # try load_delegate('edgetpu.dll')
    try :
        interpreter = Interpreter(
            model_path=model,
            experimental_delegates=[load_delegate('edgetpu.dll')])
    except OSError :
        interpreter = Interpreter(
            model_path=model)

    interpreter.allocate_tensors()

    inputDetails = interpreter.get_input_details()[0]
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

        detectImage(
            image= image,
            interpreter= interpreter,
            inputDetails= inputDetails,
            imageSize= imageSize,
            thres= thres,
            labels= labels
        )
        
        # timer end
        endTime = timeit.default_timer()
        evaluedFps = int(1./(endTime - frameStartTime))
        print(f"evaluedFps : {evaluedFps}")

        if cv2.waitKey(1) == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()



main()
