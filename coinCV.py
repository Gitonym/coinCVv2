import argparse
from asyncio.windows_events import NULL
from bisect import bisect_left
from fileinput import filename
from multiprocessing.sharedctypes import Value
from os import getcwd
from tkinter import *
from tkinter import filedialog as fd

import cv2 as cv
import imutils
import matplotlib.pyplot as pp
import numpy as np
import PIL.Image
import PIL.ImageTk
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from imutils import contours, perspective
from matplotlib.pyplot import grid
from PIL import Image, ImageTk
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance as dist
from skimage import feature
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils

current_image = None
current_canvas = None
scanned_canvas = None
detected_canvas = None
displayed_image = None
current_filename = None
scanned_image = None
detected_image = None
#rgb colors for copper gold and silver
rgbGold = sRGBColor(196/255, 179/255, 133/255)
rgbCopper = sRGBColor(193/255, 159/255, 90/255)
rgbSilver = sRGBColor(177/255, 161/255, 139/255)
#lab colorspace variants for copper, gold and silver
labGold = convert_color(rgbGold, LabColor)
labCopper = convert_color(rgbCopper, LabColor)
labSilver = convert_color(rgbSilver, LabColor)
#calibration coin
calibration_coin = None
pixelsPerMetric = None
coin_diameters = {
  1 : 16.25,
  2 : 18.75,
  5 : 21.75,
  10 : 19.75,
  20 : 22.25,
  50 : 24.25,
  100 : 23.25,
  200 : 25.75
}

def get_sizetype(diameter):
  global coin_diameters
  current_diameter = None
  sizetype = None
  
  for st in coin_diameters:
    if sizetype is None:
      current_diameter = coin_diameters[st]
      sizetype = st
      
    if abs(coin_diameters[st] - diameter) < abs(current_diameter - diameter):
      current_diameter = coin_diameters[st]
      sizetype = st
      
  return sizetype
  
def create_window():
  global current_canvas
  global scanned_canvas
  global detected_canvas
  global calibration_coin
  
  #choose File
  Button(root, text='Choose File', command=select_file).grid(row=0, column=0, sticky='ne')
  
  #calibration Coin
  Label(root, text="Top Left Coin").grid(row=1, column=0, sticky='ne')
  calibration_coin = Text(root, height=1, width=5)
  calibration_coin.grid(row=1, column=1, sticky='nw')
  
  #scan image
  Button(root, text='Scan Image', command=scan_image_watershed).grid(row=1, column=0, sticky='se')
  
  #current canvas
  current_canvas = Canvas(root, width=500, height=500)
  current_canvas.grid(row=0, column=2, rowspan=2)
  
  #scanned canvas
  scanned_canvas = Canvas(root, width=500, height=500)
  scanned_canvas.grid(row=0, column=3, rowspan=2)
  
  #detected canvas
  detected_canvas = Canvas(root, width=500, height=500)
  detected_canvas.grid(row=0, column=4, rowspan=2)
  
def scan_image_hough():
  global current_filename
  global scanned_canvas
  global scanned_image
  global detected_canvas
  global detected_image
  global coin_diameters
  global calibration_coin
  global pixelsPerMetric
  
  #delete previous calibration
  pixelsPerMetric = None
  coin_contours = []
  coin_masks = []
  
  #open image
  image = cv.imread(current_filename)
  original = image.copy()
  
  ######################- Prepare Image -####################
  #greyscale
  image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
  #blur
  image = cv.blur(image, (5, 5))
  #edges
  ret, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  #invert so that the coins are white
  print('Color: ', image[0][0])
  if image[0][0] == 255:
    image = cv.bitwise_not(image)
    print('INVERTED')
  else:
    print('NOT INVERTED')
    
  #cleanup removes the clutter
  image = cv.erode(image, None, iterations=10)
  image = cv.dilate(image, None, iterations=10)
  #cleanup merges the coins
  image = cv.dilate(image, None, iterations=8)
  image = cv.erode(image, None, iterations=8)
  
  ##########- Remove bad Contours -###########
  #get contours
  cnts = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  # sort the contours from left-to-right and initialize the
  (cnts, _) = contours.sort_contours(cnts)
  #iterate over contours
  for c in cnts:
    #iterate over coordinates in contour
    for coords in c:
      x = coords[0][0]
      y = coords[0][1]
      #if a contour touches the edge
      if x <= 0 or y <= 0 or x >= image.shape[1]-1 or y >= image.shape[0]-1:
        #create a mask for the contour
        mask = np.zeros(image.shape[:2], dtype=image.dtype)
        cv.drawContours(mask, [c], 0, (255), -1)
        mask = cv.bitwise_not(mask)
        #remove the mask
        image = cv.bitwise_and(image, image, mask=mask)
        
  #Hough
  circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=40, param1=100, param2=30, minRadius=40)
  
  #draw circles
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(original, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(original, center, radius, (255, 0, 255), 3)
      
  #display scanned
  cv.imwrite(getcwd() + '\\temp.jpg', image)
  image = Image.open(getcwd() + '\\temp.jpg')
  image = fit_image(image)
  scanned_image = ImageTk.PhotoImage(image)
  scanned_canvas.create_image(0, 0, anchor=NW, image=scanned_image)
  
  #display detected
  cv.imwrite(getcwd() + '\\temp.jpg', original)
  original = Image.open(getcwd() + '\\temp.jpg')
  original = fit_image(original)
  detected_image = ImageTk.PhotoImage(original)
  detected_canvas.create_image(0, 0, anchor=NW, image=detected_image)
  
def get_current_color(color):
  color = sRGBColor(color[0]/255, color[1]/255, color[2]/255)
  color = convert_color(color, LabColor)
  
  delta_gold = delta_e_cie2000(color, labGold)
  delta_copper = delta_e_cie2000(color, labCopper)
  delta_silver = delta_e_cie2000(color, labSilver)
  
  if delta_gold < delta_copper and delta_gold < delta_silver:
    return "Gold"
  
  if delta_copper < delta_gold and delta_copper < delta_silver:
    return "Copper"
  
  if delta_silver < delta_copper and delta_silver < delta_gold:
    return "Silver"
  
  return "Keine Farbe"
  
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def scan_image_watershed():
  global current_filename
  global scanned_canvas
  global scanned_image
  global detected_canvas
  global detected_image
  global coin_diameters
  global calibration_coin
  global pixelsPerMetric
  
  #delete previous calibration
  pixelsPerMetric = None
  coin_contours = []
  coin_masks = []
  
  #open image
  image = cv.imread(current_filename)
  original = image.copy()
  
  ######################- Prepare Image -####################
  #greyscale
  image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
  #blur
  image = cv.blur(image, (5, 5))
  #edges
  ret, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  #invert so that the coins are white
  print('Color: ', image[0][0])
  if image[0][0] == 255:
    image = cv.bitwise_not(image)
    print('INVERTED')
  else:
    print('NOT INVERTED')
    
  #cleanup removes the clutter
  image = cv.erode(image, None, iterations=10)
  image = cv.dilate(image, None, iterations=10)
  #cleanup merges the coins
  image = cv.dilate(image, None, iterations=8)
  image = cv.erode(image, None, iterations=8)
  
  ##########- Remove bad Contours -###########
  #get contours
  cnts = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  # sort the contours from left-to-right and initialize the
  (cnts, _) = contours.sort_contours(cnts)
  #iterate over contours
  for c in cnts:
    #iterate over coordinates in contour
    for coords in c:
      x = coords[0][0]
      y = coords[0][1]
      #if a contour touches the edge
      if x <= 0 or y <= 0 or x >= image.shape[1]-1 or y >= image.shape[0]-1:
        #create a mask for the contour
        mask = np.zeros(image.shape[:2], dtype=image.dtype)
        cv.drawContours(mask, [c], 0, (255), -1)
        mask = cv.bitwise_not(mask)
        #remove the mask
        image = cv.bitwise_and(image, image, mask=mask)
  
  ##########- WATERSHED -################
  #distance transform
  shifted = cv.pyrMeanShiftFiltering(original, 21, 51)
  gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
  D = ndimage.distance_transform_edt(image)
  localMax = peak_local_max(D, indices=False, min_distance=60, labels=image)
  markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
  labels = watershed(-D, markers, mask=image)
  
  #for every unique coin
  for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
      continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    #save the coin mask
    coin_masks.append(mask)
    # detect contours in the mask and grab the largest one
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    coin_contour = max(cnts, key=cv.contourArea)
    #add the contour to the list of contours
    coin_contours.append(coin_contour)
  #sort the contours from elft to right
  (coin_contours, _) = contours.sort_contours(coin_contours)
  iteration = 0
  #for every contour
  for c in coin_contours:
    iteration += 1
    # draw a rectangle enclosing the object
    box = cv.minAreaRect(c)
    box = cv.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv.drawContours(original, [box.astype("int")], -1, (0, 255, 0), 2)
    #calculate diameter
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    diameter = (dA + dB)/2
    
    if diameter > 80:
      #############- Color -###############
      color_mask = coin_masks[iteration-1]
      cv.erode(color_mask, None, iterations=8)
      average_color = cv.mean(original, mask=color_mask)
      average_color = (int(average_color[0]), int(average_color[1]), int(average_color[2]))
      average_color_rgb = (average_color[2], average_color[1], average_color[0])
      
      ################- Calibrate -##############
      if pixelsPerMetric is None:
        pixelsPerMetric = diameter / coin_diameters[int(calibration_coin.get("1.0", "end"))]

      #############- Draw Information -############
      print("#{}".format(iteration), '\t Color: ', average_color_rgb , "\t Diameter: " , diameter/pixelsPerMetric, 'mm', "\t Colortype: ", get_current_color(average_color_rgb), '\t Sizetype: ', get_sizetype(diameter/pixelsPerMetric))
      x = tl[0] + (br[0] - tl[0])//2
      y = tl[1] + (br[1] - tl[1])//2
      cv.putText(original, "#{}".format(iteration), (int(x) - 30, int(y) + 10), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
      
  #display scanned
  cv.imwrite(getcwd() + '\\temp.jpg', image)
  image = Image.open(getcwd() + '\\temp.jpg')
  image = fit_image(image)
  scanned_image = ImageTk.PhotoImage(image)
  scanned_canvas.create_image(0, 0, anchor=NW, image=scanned_image)
  
  #display detected
  cv.imwrite(getcwd() + '\\temp.jpg', original)
  original = Image.open(getcwd() + '\\temp.jpg')
  original = fit_image(original)
  detected_image = ImageTk.PhotoImage(original)
  detected_canvas.create_image(0, 0, anchor=NW, image=detected_image)
  
def display_image(image):
  global current_canvas
  global displayed_image
  
  image = fit_image(image)
  displayed_image = ImageTk.PhotoImage(image)
  current_canvas.create_image(0, 0, anchor=NW, image=displayed_image)
  
def fit_image(image):
  width = int(image.size[0])
  height = int(image.size[1])
  
  if width > height:
    ratio = 500 / width
  else:
    ratio = 500 / height
    
  return image.resize((int(width * ratio), int(height * ratio)))
  
def select_file():
  global current_canvas
  global current_image
  global current_filename
  
  filetypes = (
    ('image files', '*.jpg'),
  )
  current_filename = fd.askopenfilename(initialdir=getcwd() + '\\examples', filetypes=filetypes)
  root.title("coinCV: " + current_filename)
  current_image = cv.imread(current_filename,0)
  
  display_image(Image.open(current_filename))

root = Tk()
root.title("CoinCV: No File Chosen")
root.resizable(False, False)
create_window()
root.mainloop()
