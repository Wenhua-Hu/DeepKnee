import numpy as np 
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import cv2
import imutils

import matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dropout, Input, Lambda, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import imagenet_utils

from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input 
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input 
# from tensorflow.python.keras.applications.efficientnet import EfficientNetB6, preprocess_input 
import matplotlib


from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
#disable_eager_execution()

################################################################################################


#@title Cam Class
####
class GradCAM:
  def __init__(self, model, classIdx, layerName=None):
    # store the model, the class index used to measure the class
    # activation map, and the layer to be used when visualizing
    # the class activation map
    self.model = model
    self.classIdx = classIdx
    self.layerName = layerName
    # if the layer name is None, attempt to automatically find
    # the target output layer
    if self.layerName is None:
      self.layerName = self.find_target_layer()
   
  def find_target_layer(self):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(self.model.layers):
      # check to see if the layer has a 4D output
      if len(layer.output_shape) == 4:
        return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
  
  def compute_heatmap(self, image, eps=1e-8):
    # construct our gradient model by supplying (1) the inputs
    # to our pre-trained model, (2) the output of the (presumably)
    # final 4D layer in the network, and (3) the output of the
    # softmax activations from the model
    gradModel = Model(
      inputs=[self.model.inputs],
      outputs=[self.model.get_layer(self.layerName).output,
        self.model.output])
  
  	# record operations for automatic differentiation
    with tf.GradientTape() as tape:
      # cast the image tensor to a float-32 data type, pass the
      # image through the gradient model, and grab the loss
      # associated with the specific class index
      inputs = tf.cast(image, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, self.classIdx]
      # use automatic differentiation to compute the gradients
      grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap


  def overlay_heatmap(self, heatmap, image, alpha=0.5,
    colormap=cv2.COLORMAP_VIRIDIS): ##### does it support to receive the color map from matplotlib , there is need to investigate color map using from same lib
    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    heatmap = cv2.applyColorMap(heatmap, colormap)  #### cv2 or matplotlib or other lib
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)   #### for loverlay function  with weight??
    # return a 2-tuple of the color mapped heatmap and the output,
    # overlaid image
    return (heatmap, output)


def load_image(path_to_image):
    orig = cv2.imread(path_to_image)
    resized = cv2.resize(orig, (224, 224))
    
    image = load_img(path_to_image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)   #############  why do we add 1 more dimension =>  4 dimension ??
    return orig, imagenet_utils.preprocess_input(image)   ######### ?

    
def predict_on_image(image):

    orig = cv2.imread(IMG_PATH)
    resized = cv2.resize(orig, (224, 224))
    
    image = load_img(IMG_PATH, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    labels = [0,1,2,3,4]
    preds = model.predict(image)
    i = np.argmax(preds[0])
    label = labels[i]
    percentage = preds[0][np.argmax(preds)]*100
    return (label, percentage)

def get_heatmap(model, image, label, orig):
    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, label)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    return cam.overlay_heatmap(heatmap, orig, alpha=0.5)

class GroupM():
  """Class that keeps track of one Area of Interest which is found by looking
  at the pixels in a heatmap that are above a certrain thresholds. Using the
  functions from the Groupmatrix class.
  
  One needs to initalize the class by calling the class with one coördinate"""
  def __init__(self, cord):
    self.dic = {} # dictionary that stores the coördinates of pixels belonging to this class
    self.group_size = 0

    self.dic[cord[0]] = [cord[1]] # initialize dictionary
    self.top = None
    self.left = None
    self.right = None
    self.bot = None
    self.perc_int = None


  def want_to_add(self, cord):
    # Function to determine if a new coördinate should be part of this class
    # Input: a coördinate of a pixel
    x,y = cord
    
    #since the 'find_group' algorithm from the Groupmatrix class goes from top left
    # to bottom right we only need to considers pixels above and left of the input cord
    top = max(0,y-1) # max accounts for y values of 0
    left = max(0,x-1)
    if x in self.dic:
      return top in self.dic[x]
    elif left in self.dic:
      return y in self.dic[left]

    return False

    
  def add_to_group(self, cord):
    # Function to add input coördinate to this group
    # Input: a coördinate of a pixel
    x,y = cord
    if x in self.dic:
      self.dic[x].append(y)
    else:
      self.dic[x] = [y]

  def merge_groups(self, group):
    # Function to add another group to this one
    # Input another group of pixels of type: GroupM
    for key in group.dic:
      if key in self.dic:
        self.dic[key] = self.dic[key] + group.dic[key]
      else:
        self.dic[key] = group.dic[key]

  def get_group_size(self):
    # Function to get the total size of the group. While calling this function
    # the edges of the group are also stored in: self.top/bot/left/right
    tot = 0
    top= 0
    bot = 10000 # large constant
    left = 10000
    right = 0

    for key in self.dic:
      #a key in the self.dic represents the y value of a coördinate
      if key < bot:
        bot = key
      if key > top:
        top = key
      cols = self.dic[key]

      #looks at the smallest and highest x coördinate of this row of cords
      comp_left = np.min(cols)
      comp_right = np.max(cols)
    
      if comp_left < left:
        left = comp_left
      if comp_right > right:
        right = comp_right
      tot += len(cols)
      
    self.group_size = tot
    self.top = top
    self.bot = bot
    self.right = right
    self.left = left
    return tot

  def get_edges(self):
    # returns edges of this group
    return self.top, self.bot, self.left, self.right

  def perc_intensity(self, heatmap):
    # Function to return the average intensity of the pixels belonging to this group
    # Takes the heatmap used to create this group as an input
    tot_intensity = np.sum(heatmap)
    tot_group_intensity = 0
    for x in self.dic:
      ys = self.dic[x]
      for y in ys:
        tot_group_intensity += heatmap[x,y]
    percentage_int = tot_group_intensity / tot_intensity
    self.perc_int = percentage_int
    return percentage_int

class Groupmatrix(object):
  """ Class that takes a heatmap represented as a numpy array (matrix)
  and a intensity treshold and locates groups of pixels that are above th treshold"""

  def __init__(self, heatmap, intensity = 100):
    self.heatmap = heatmap
    self.intensity = intensity
    self.groups = []
    self.mat = heatmap > self.intensity # transform matrix into binaries (False,True)


  def find_groups(self):
    nonz = np.nonzero(self.mat)
    # iterate over nonzero elements in matrix
    for i in range(len(nonz[0])):
      x= nonz[0][i]
      y = nonz[1][i]
      cord = [x,y]
      in_group_list = []

      # find groups to add current coördinate to
      for j,group in enumerate(self.groups):
        if group.want_to_add(cord):
          in_group_list.append(j)
      

      k = len(in_group_list)

      # if no groups were found to add current cord to add a new group
      if k == 0:
        self.groups.append(GroupM(cord))
      
      # if only one group was found to add current cord to add it to the group
      elif k == 1:
        group = self.groups[in_group_list[0]]
        group.add_to_group(cord)
        self.groups[in_group_list[0]] = group

      # if multiple groups were found this cord now functions as a connector
      # between those groups and thus we can merge the groups and add the cord 
      # itself after
      else:
        main_group = self.groups[in_group_list[0]]
        while k > 1:
          subgroup = self.groups[in_group_list[k-1]]
          main_group.merge_groups(subgroup)
          del self.groups[in_group_list[k-1]]
          k -=1
          main_group.add_to_group(cord)
          
          self.groups[in_group_list[0]] = main_group

  def return_groups(self, n = 625, min_intens = 0.05):
    # Function to return all groups that have at least n members
    out_groups = []
    for group in self.groups:
      group_size = group.get_group_size()
      perc_int = group.perc_intensity(self.heatmap)
      if group_size >= n and perc_int >= min_intens:
        out_groups.append(group)

    return out_groups
  
class HeatmapSquares(object):
  """Class that takes an image (224, 224, x) and a list of groups (type GroupM) 
  of pixels that were found using the GroupMatrix class. This class is used to
  draw squares around the groups, which represent areas of interest."""
  def __init__(self, image, group_list):
    self.image = image
    self.group_list = group_list
    self.cmap = None

  @staticmethod
  def create_cmap(bot, top):
    top = matplotlib.cm.get_cmap(top, 128)
    bottom = matplotlib.cm.get_cmap(bot, 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                          bottom(np.linspace(0, 1, 128))))
    return matplotlib.colors.ListedColormap(newcolors)

  @staticmethod
  def draw_one_group(image, group, line_thickness, line_color):
    # Function to draw a rectangle around one group

    cloned_image = image.copy() # copy the image so that the original does not get polluted
    top,bot,left,right = group.get_edges()
    cv2.line(cloned_image, (right,bot), (left,bot), line_color, line_thickness)
    cv2.line(cloned_image, (right,top), (left,top), line_color, line_thickness)
    cv2.line(cloned_image, (left,top), (left,bot), line_color, line_thickness)
    cv2.line(cloned_image, (right,top), (right,bot), line_color, line_thickness)

    return cloned_image
  
  def draw_all_groups(self, intensity_based_size = False, accuracy_color = False,
                     color_map = 'Greens'):
    
    line_thickness = 3
    line_color = (255,0,0)
    # ==========================================================
    # if accuracy_color:
    #   acc = 1 - round(accuracy_color, 2)
    #   print(acc)
    #   rgba = color_map   # color_map(acc)       # ----------------   ?  function or string
    #   rgb_value= np.asarray(matplotlib.colors.to_rgb(rgba)) * 255   #-------------- something wrong
    #   print(rgb_value)
    #
    #   r, g, b = 100
    #   line_color = (r,g,b)


    # ============================================================

    line_color = (100, 200 , 150)
    # Function to draw all rectangles around all groups on an image
    n_groups = len(self.group_list)
    # If the grouplist is empty the original image is returned
    if n_groups == 0:
      return self.image.copy()
    else:
      group = self.group_list[0]

      if intensity_based_size:
        perc_intens = group.perc_int
        line_thickness = max(1,int(round(perc_intens,1) * 10))

      cloned = self.draw_one_group(self.image, group, 
                                   line_thickness,line_color)
      for group in self.group_list[1:]:
        if intensity_based_size:
          perc_intens = group.perc_int
          line_thickness = max(1,int(round(perc_intens,1) * 10))

        cloned = self.draw_one_group(cloned, group, 
                                   line_thickness,line_color)
      return cloned

def draw_squares(model, output, image, orig, min_intensity = 120,
                 min_size = 625, min_perc = 0.03,
                 cmap='inferno',
                 intensity_based_size = False, accuracy = 0,):
    """Function to quickly draw squares around area of interest in radio-image
    uses HeatMapSquares, GroupsM and GroupMatrix classes as well as the heatmap
    for finding intensity of pixels
    model - prediction model
    output - predicted class of image
    image - resized image used for prediction
    orig - original unchanged image
    min_intensity - minimum intensity required to become part of group [0-255]
    min_size = minimum size of group needed to have a square be drawn
    min_perc = minimum contribution requirement of a group to total intensity of an image
    cmap - matplotlib color map
    """
    
    # Find groups
    cam = GradCAM(model, output)
    heatmap = cam.compute_heatmap(image)
    group_class = Groupmatrix(heatmap, min_intensity)
    group_class.find_groups()
    out_groups = group_class.return_groups(min_size,min_perc)
    
    # Draw groups
    hs = HeatmapSquares(orig, out_groups)
    output_image = hs.draw_all_groups(intensity_based_size, accuracy,cmap)
    
    return output_image
        

if __name__ == "__main__":
    SHOW = True

    #path to example image
    IMG_PATH = r'/UVA21_DSP_QUIN/images/9001400L.png'
    #path to stored model: /VGG16-acc49 (whole folder with .pb file in it)
    MODEL_PATH = r'/UVA21_DSP_QUIN/data/models/VGG16-acc49'
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])   # no clue 1

    
    #LOAD MODEL: 
    model = keras.models.load_model(MODEL_PATH) 
    
    #load resized and original image
    orig, image = load_image(IMG_PATH)
    
    #get model prediction and percentage 
    label, percentage = predict_on_image(image)
    
    #get image with squares
    #variables to change:
        
        #min_intensity
        #min_size
        #min_perc
        #min_cmap
        #intensity_based
    squares = draw_squares(model, label, image, orig, intensity_based_size=True,accuracy=percentage)  # guess  it is the point
    #print(squares)
    
    # if SHOW:
    #cv2.imshow(squares)
    
    #get heatmap picture and heatmap picture on top of original
    #heatmap, output = get_heatmap(model, image, label, orig)
    # Window name in which image is displayed
    window_name = 'image'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, squares)

    #waits for user to press any key
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    #closing all open windows
    cv2.destroyAllWindows()