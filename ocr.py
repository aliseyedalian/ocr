import cv2
import pytesseract

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

def get_grayscale(img):
    grayscaleimage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return grayscaleimage

def thresholding(img):
    return cv2.threshold(img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def remove_noise(img):
    return cv2.medianBlur(img,6)

def show(img,sec,title='image'):
    cv2.imshow(title,img)
    cv2.waitKey(1000*sec)

img = cv2.imread('image.jpg')
show(img,1)
img = get_grayscale(img)
show(img,1,title="after get_grayscale")
img = thresholding(img)
show(img,1,title="after thresholding")
img = remove_noise(img)
show(img,1,title="after remove_noise")
text = ocr_core(img)

print(text)
