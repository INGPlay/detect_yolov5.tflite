import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    model = 'yolov5n-fp16-320.tflite'
    interpreter = tf.lite.Interpreter(model_path=model)
    # interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # input
    inputDetails = interpreter.get_input_details()[0]

    image = cv2.imread('image4.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # crop
    imageShape = image.shape
    originalWidth = int(imageShape[0])
    originalHeight = int(imageShape[1])
    sliX = int(originalWidth/2 - 160)
    sliY = int(originalHeight/2 - 160)

    imageSize = 320
    image = image[sliX : sliX + imageSize, sliY : sliY + imageSize]

    # normalize
    imageNor = [np.float32(cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX))]
    interpreter.set_tensor(inputDetails['index'], imageNor)

    interpreter.invoke()

    # ouput
    outputDetails = interpreter.get_output_details()[0]['index']
    output = interpreter.get_tensor(outputDetails)[0]

    labels = []
    with open("labels.txt", "r") as fs :
        lines = fs.readlines()
        for l in lines :
            l = l.strip()
            labels.append(l)

    thres = 0.5
    for i in range(len(output)) :
        conf = output[i][4]

        if conf >= thres :
            w = round(output[i][2] * imageSize)
            h = round(output[i][3] * imageSize)
            x = round(output[i][0] * imageSize-w*0.5)
            y = round(output[i][1] * imageSize-h*0.5)

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


if __name__ == "__main__" :
    main()