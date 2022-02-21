# try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
from tflite_runtime.interpreter import Interpreter, load_delegate
# except ImportError:
#     import tensorflow as tf
#     Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
import numpy as np
import cv2
from tools.editImage import cropImageCenter
from tools.editImage import draw_text


def detectImage(image, interpreter, inputDetails, outputDetails, imageSize, thres, labels) :

    image = cropImageCenter(image=image, width=imageSize, height=imageSize)
    # plt.imshow(image); plt.show()

    # quantinzation
    scale, zeroPoint = inputDetails["quantization"]

    qImage = (image / scale + zeroPoint).astype(np.uint8)
    #input
    interpreter.set_tensor(inputDetails['index'], [qImage])

    interpreter.invoke()

    # output
    output = interpreter.get_tensor(outputDetails['index'])[0]
    scale, zeroPoint = outputDetails["quantization"]
    output = (output.astype(np.float32) - zeroPoint) * scale

    for i in range(len(output)) :
        conf = output[i][4]
        
        if conf >= thres :
            w = round(output[i][2] * imageSize)
            h = round(output[i][3] * imageSize)
            x = round(output[i][0] * imageSize - 0.5 * w)
            y = round(output[i][1] * imageSize - 0.5 * h)

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
    return image
