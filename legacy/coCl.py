import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import numpy as np
import cv2
import os
import sys
import timeit
import matplotlib.pyplot as plt


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


def main() :

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    imageSize = 320
    image = cv2.imread('image.png')
    model = 'yolov5n-int8-320_edgetpu.tflite'
    interpreter = tf.lite.Interpreter(
        model_path=model)
    # interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()

    inputDetails = interpreter.get_input_details()[0]

    # Label load
    fs = open("labels.txt", "r")
    labels = []
    lines = fs.readlines()
    for l in lines :
        l = l.strip()
        labels.append(l)

    fs.close()

    image = cropImageCenter(image=image, width=imageSize, height=imageSize)
    # plt.imshow(image); plt.show()

    #normalize
    imageNor = [np.uint8(cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX))]
    interpreter.set_tensor(inputDetails['index'], np.uint8(imageNor))

    interpreter.invoke()

    # output
    outputDetails = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(outputDetails['index'])[0]

    for i in range(len(output)) :
        print(f"0 : {output[i][0]}, 1 : {output[i][1]}, 2 : {output[i][2]}, 3 : {output[i][3]}, 4 : {output[i][4]}, 5 : {output[i][5:].argmax()}")
        conf = output[i][4]

        if conf >= 50 :
            w = round(output[i][2]-32)
            h = round(output[i][3]-32)
            x = round(output[i][0]-w*0.5)
            y = round(output[i][1]-h*0.5)

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

    plt.imshow(image); plt.show()

        
    cv2.destroyAllWindows()



main()
