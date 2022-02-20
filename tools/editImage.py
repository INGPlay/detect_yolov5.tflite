import cv2

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
          font_scale=2,
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