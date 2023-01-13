import random
import argparse
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import math

# center window function. def center_window(w=300, h=200):
def center_window(w=300, h=200):
    # get screen width and height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    # calculate position x, y
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

# contrast border thumbnail
root = Tk()
root.title("Đỗ Hoàng Phúc - 4501104178")

mainphoto = ImageTk.PhotoImage(file="background.jpg")
background_label = Label(root, image=mainphoto)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#root.configure(background='cyan')


center_window(1130, 460)


def openfn():
    filename = filedialog.askopenfilename(title='Places')
    return filename

# function of btn

def selected():
    global path, img
    path = openfn()
    image = cv2.imread(path)
    img = Image.fromarray(image)
    # img.thumbnail((350, 350))
    img = ImageTk.PhotoImage(img)
    canvasO.create_image(175, 215, image=img)
    canvasO.image = img

def Resized():
    global path
    image = cv2.imread(path)
    resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(resized)
    # img.thumbnail((350, 350))
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def canny():
    global path
    image = cv2.imread(path, 0)
    edges = cv2.Canny(image, 100, 200)  # Image, min and max values
    img = Image.fromarray(edges)
    # img.thumbnail((350, 350))
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def addValue():
    global path
    value = add.get()
    convert = int(value)
    image = cv2.imread(path, 1)
    img1 = cv2.add(image, convert)
    img = Image.fromarray(img1)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def subtractValue():
    global path
    value = subtract.get()
    convert = int(value)
    image = cv2.imread(path, 1)
    img1 = cv2.subtract(image, convert)
    img = Image.fromarray(img1)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def multiplyValue():
    global path
    value = multiply.get()
    convert = int(value)
    image = cv2.imread(path, 1)
    img1 = cv2.multiply(image, convert)
    img = Image.fromarray(img1)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def logFormulas():
    global path
    image = cv2.imread(path, 0)

    img_1 = np.uint8(np.log1p(image))
    thresh = 3
    img_2 = cv2.threshold(img_1, thresh, 255, cv2.THRESH_BINARY)[1]

    img = Image.fromarray(image)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def powerFormulas():
    global path
    image = cv2.imread(path, 0)
    img_1 = np.uint8(np.log1p(image))
    gamma = 0.4
    img_2 = np.power(img_1, 1/gamma)
    img = Image.fromarray(img_2)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def filterDirections():
    global path
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    kernelsx = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    kernelsy = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], ])
    # rotated 90 degrees
    kernelsm = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    kernelsn = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    img_prewx = cv2.filter2D(img_gaussian, -1, kernelsx / 3)
    img_prewy = cv2.filter2D(img_gaussian, -1, kernelsy / 3)
    img_prewm = cv2.filter2D(img_gaussian, -1, kernelsm / 3)
    img_prewn = cv2.filter2D(img_gaussian, -1, kernelsn / 3)
    sum = img_prewx + img_prewy + img_prewm + img_prewn
    img = Image.fromarray(sum)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def gamma():
    global path
    value = float(valuesGam1.get())
    image = cv2.imread(path)
    gamma_corrected = np.array(255 * (image / 255) ** value, dtype='uint8')
    img = Image.fromarray(gamma_corrected)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def manualQuantization():
    global path
    value = float(valuesGam.get())
    image = cv2.imread(path)
    numOfLevel = 2. ** value
    levelGap = 256 / numOfLevel
    quantizedImg = np.uint8(image / levelGap) * levelGap - 1
    img = Image.fromarray((quantizedImg * 255).astype(np.uint8))
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def sp_noise(image, prob):
    # Add salt and pepper noise to image
    # prob: Probability of the noise
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def noise():
    global path
    value = float(enNoise.get())
    image = cv2.imread(path, 1)
    noise_img = np.zeros(image.shape, dtype=np.uint8)
    cv2.randn(noise_img, 0, value)
    new_img = image + noise_img

    img = Image.fromarray(new_img)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def MedianFilter():
    image = cv2.imread(path, 1)
    noise_img = sp_noise(image, 0.05)
    #noise_img = cv2.resize(noise_img, (225, 255))
    img = Image.fromarray(noise_img)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def gkern(l, sig):
    ax = np.linspace(-(l-1)/2.,(l-1)/2.,l)
    gauss = np.exp(-0.5*np.square(ax)/np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def gaussfilters():
    global path
    size = int(lCombo.get())
    sigma = int(valuesSig.get())
    value = gkern(size, sigma)
    image = cv2.imread(path)
    img1 = cv2.filter2D(image, -1, value)
    img = Image.fromarray(img1)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def leftrotate():
    global path
    image = cv2.imread(path, 1)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    #val = 15
    rotate_matrix = cv2.getRotationMatrix2D(center, 20, 1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    img = Image.fromarray(rotated_image)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def rightrotate():
    global path
    image = cv2.imread(path, 1)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotate_matrix = cv2.getRotationMatrix2D(center, -20, 1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    img = Image.fromarray(rotated_image)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def halfround():
    global path
    image = cv2.imread(path, 1)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotate_matrix = cv2.getRotationMatrix2D(center, 180, 1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    img = Image.fromarray(rotated_image)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def binary():
    global path
    image = cv2.imread(path, 1)

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    img = Image.fromarray(blackAndWhiteImage)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

def Bilateralfilter():
    global path
    image = cv2.imread(path, 1)
    bila = cv2.bilateralFilter(image, 25, 80, 100)
    img = Image.fromarray(bila)
    img = ImageTk.PhotoImage(img)
    canvasR.create_image(175, 215, image=img)
    canvasR.image = img

# create input
add = ttk.Entry(root, font=('ariel 10 bold'), width="20")
#add.place(x=50, y=9)
add.place(x=200, y=10)
add.bind("<<add>>", addValue)

subtract = ttk.Entry(root, font=('ariel 10 bold'), width="20")
subtract.place(x=200, y=53)
subtract.bind("<<subtract>>", subtract)

multiply = ttk.Entry(root, font=('ariel 10 bold'), width="20")
#multiply.place(x=75, y=85)
multiply.place(x=200, y=96)
multiply.bind("<<multiply>>", multiply)

enNoise = ttk.Entry(root, font=('ariel 10 bold'), width="20")
#enNoise.place(x=60, y=45)
enNoise.place(x=200, y=190)
enNoise.bind("<<noise>>", noise)

# create combobox
valuel = [1, 3, 5, 7, 9, 11, 13, 17, 17, 19, 21]
lCombo = ttk.Combobox(root, values=valuel, font=('ariel 10'), state="readonly", width="8")
lCombo.place(x=269, y=235)
lCombo.bind("<<ComboboxSelected>>", lCombo)
lCombo.set('size')

valuesSig = [ 1, 2, 3, 4, 5, 6, 7, 8, 9]
valuesSig = ttk.Combobox(root, values=valuesSig, font=('ariel 10'), state="readonly", width="7")
valuesSig.place(x=200, y=235)
valuesSig.bind("<<ComboboxSelected>>", valuesSig)
valuesSig.set('sigma')

valuesGam1 = ttk.Combobox(root, values=valuel, font=('ariel 10'), state="readonly", width="18")
valuesGam1.place(x=200, y=139)
valuesGam1.bind("<<ComboboxSelected>>", valuesGam1)
valuesGam1.set('gamma')

valuesGam = ttk.Combobox(root, values=valuel, font=('ariel 10'), state="readonly", width="18")
valuesGam.place(x=200, y = 366)
valuesGam.bind("<<ComboboxSelected>>", valuesGam)
valuesGam.set('gamma')

#menu
menubar = Menu(root)
file = Menu(menubar, tearoff=0)

thisphoto = ImageTk.PhotoImage(file="Canny1.jpg")
thisphoto1 = ImageTk.PhotoImage(file="LRotate.jpg")
thisphoto2 = ImageTk.PhotoImage(file="RRotate.jpg")
thisphoto3 = ImageTk.PhotoImage(file="halfRotate.jpg")
thisphoto4 = ImageTk.PhotoImage(file="Binary.png")

menubar.add_command(label="Select", command=selected)
menubar.add_command(label="Resized", command=Resized)

# menubar.add_command(label="Median Filter", command=MedianFilter)
# menubar.add_command(label="Power Formulas", command=powerFormulas)
# menubar.add_command(label="Log Formulas", command=logFormulas)
menubar.add_command(label="Bilateral filter", command=Bilateralfilter)


file.add_command(image=thisphoto4, label="Binary", command=binary, compound=LEFT)
file.add_command(image=thisphoto, label="Canny", command=canny, compound=LEFT)
file.add_command(image=thisphoto1, label="Left Rotate", command=leftrotate, compound=LEFT)
file.add_command(image=thisphoto2, label="Right Rotate", command=rightrotate, compound=LEFT)
file.add_command(image=thisphoto3, label="180° Rotate", command=halfround, compound=LEFT)
menubar.add_cascade(label="Choise", menu=file)
root.config(menu=menubar)




# menubar.add_cascade(label="Left Rotate", command=leftrotate, compound=LEFT)
# root.config(menu=menubar)
# menubar.add_cascade(label="Right Rotate", command=rightrotate, compound=LEFT)
# root.config(menu=menubar)


#create button

#C:\Users\PC\PycharmProjects\pythonProject2\add.png
# photokaito = ImageTk.PhotoImage(file="add.png")
# photokaito1 = ImageTk.PhotoImage(file="Subtract.png")
# photokaito2 = ImageTk.PhotoImage(file="Multiply.png")
# photokaito3 = ImageTk.PhotoImage(file="Gauss.jpg")
# photokaito4 = ImageTk.PhotoImage(file="noise.jpg")
# photokaito5 = ImageTk.PhotoImage(file="Median.jpg")
# photokaito6 = ImageTk.PhotoImage(file="Direction.jpg")
# photokaito7 = ImageTk.PhotoImage(file="Gamma.jpg")
# photokaito8 = ImageTk.PhotoImage(file="LOG.jpg")
# photokaito9 = ImageTk.PhotoImage(file="powerFomulas.jpg")

#photokaito = cv2.imread("add.png")
logFormulas = Button(root, bd=5, text="    Log Formulas   ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=logFormulas, compound = LEFT)
logFormulas.grid(column = 1, row = 6, padx = 40, pady = 5, sticky = 'w')

powerFormulas = Button(root, bd=5, text="  Power Formulas ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=powerFormulas, compound = LEFT)
powerFormulas.grid(column = 0, row = 7, padx = 5, pady = 5, sticky = 'w')

filterDirections = Button(root,bd=5, text=" Filter Directional ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=filterDirections, compound = LEFT)
filterDirections.grid(column = 1, row = 7, padx = 40, pady = 5, sticky = 'w')

medianFilter = Button(root, bd=5, text="   Median Filter    ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=MedianFilter, compound = LEFT)
medianFilter.grid(column = 0, row = 6, padx = 5, pady = 5, sticky = 'w')



btnAdd = Button(root, bd=5, text="     Add      ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=addValue, compound = LEFT)
btnAdd.grid(column = 0, row = 0, padx = 5, pady = 5, sticky = 'w')

btnSubtract = Button(root, bd=5, text="  Subtract ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED  , command=subtractValue, compound = LEFT)
btnSubtract.grid(column = 0, row = 1, padx = 5, pady = 5, sticky = 'w')

btnMultiply = Button(root, bd=5, text="   Multiply  ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=multiplyValue, compound = LEFT)
btnMultiply.grid(column = 0, row = 2, padx = 5, pady = 5, sticky = 'w')

gamma = Button(root, bd=5, text=" Gamma", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=gamma, compound = LEFT)
gamma.grid(column = 0, row = 3, padx = 5, pady = 5, sticky = 'w')

cboManualQuantization = Button(root, bd=5, text="Manual quantization", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=manualQuantization)
cboManualQuantization.grid(column = 0, row = 10, padx = 5, pady = 5, sticky = 'w')


gaussfilters = Button(root, bd=5, text=" Gauss Filters", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=gaussfilters, compound = LEFT)
gaussfilters.grid(column = 0, row = 5, padx = 5, pady = 5, sticky = 'w')

Noise = Button(root, bd=5, text="   Noise  ", bg='yellow', fg='black', font=('ariel 10 bold'), relief=RAISED , command=noise, compound = LEFT)
Noise.grid(column = 0, row = 4, padx = 5, pady = 5, sticky = 'w')




# create canvas to display origin image


canvasO = Canvas(root, width="350", height="420", relief=RAISED, bd=7)
canvasO.place(x=370, y=5)

origin = ttk.Label(root, text = "The original", foreground ="black", font = ("Times New Roman", 18))
origin.place(x=500, y=10)


# create canvas to display result image
canvasR = Canvas(root, width="350", height="420", relief=RAISED, bd=7)
canvasR.place(x=740, y=5)

origin = ttk.Label(root, text="Result", foreground ="black", font = ("Times New Roman", 18))
origin.place(x=890, y=10)

root.mainloop()