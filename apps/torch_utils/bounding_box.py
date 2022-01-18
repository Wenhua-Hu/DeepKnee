# -*- coding: utf-8 -*-
import os

import numpy as np
from PIL import Image 
import matplotlib
import cv2


class GroupM(object):
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
  """ Class that takes takes a heatmap represented as a numpy array (matrix)
  and a intensity treshold and locates groups of pixels that are above th treshold"""

  def __init__(self, heatmap, intensity = 100):
    self.heatmap = self.normalize_heatmap(heatmap)
    # print("bbox:", np.max(self.heatmap), np.min(self.heatmap))
    self.intensity = intensity
    self.groups = []
    self.mat = self.heatmap > self.intensity # transform matrix into binaries (False,True)

  def normalize_heatmap(self, heatmap):
    heatmap = np.asarray(heatmap)
    if np.max(heatmap) <= 1:
        heatmap = heatmap * 255
    return heatmap


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
  def __init__(self, image_path, group_list):
    self.image = self.get_image(image_path)

    self.group_list = group_list
    self.cmap = None

  def get_image(self, image_path):
    heatmap = Image.open(image_path)
    image = heatmap.convert('RGB')
    image = np.asarray(image)

    return image

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
    im_w, im_h, _ = image.shape
    w_transform_factor = 1
    h_transform_factor = 1
    #Changed size of 224 -> 299
    if im_w != 299:
        w_transform_factor = im_w / 299
    if im_h != 299:
        h_transform_factor = im_h / 299
        
    
    # Function to draw a rectangle around one group

    cloned_image = image.copy() # copy the image so that the original does not get polluted
    top,bot,left,right = group.get_edges()

    if w_transform_factor != 1:
        left = round(left * w_transform_factor)
        right = round(right * w_transform_factor)
        
    if h_transform_factor != 1:
        top = round(top * h_transform_factor)
        bot = round(bot * h_transform_factor)
    
    cloned_image = image.copy() # copy the image so that the original does not get polluted
    top,bot,left,right = group.get_edges()
    cv2.line(cloned_image, (right,bot), (left,bot), line_color, line_thickness)
    cv2.line(cloned_image, (right,top), (left,top), line_color, line_thickness)
    cv2.line(cloned_image, (left,top), (left,bot), line_color, line_thickness)
    cv2.line(cloned_image, (right,top), (right,bot), line_color, line_thickness)

    return cloned_image
  
  def draw_all_groups(self, intensity_based_size = False, accuracy_color = False,
                     color_map = 'green'):

    line_thickness = 3
    line_color = (255,0,0)
    # if accuracy_color:
    #   acc = 1 - round(accuracy_color, 2)
    #   rgba = color_map(acc)
    #   r,g,b = np.asarray(matplotlib.colors.to_rgb(rgba)) * 255
    #   line_color = (r,g,b)

    line_color = (100, 200, 150)
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


def draw_boundingbox(image_gradcam, image_path, outdir):
  group_class = Groupmatrix(image_gradcam, 140)
  # print("draw_boundingbox:",np.min(group_class.heatmap), np.max(group_class.heatmap), np.sum(group_class.mat))
  group_class.find_groups()

  out_groups = group_class.return_groups(625, 0.03)

  # Draw groups
  hs = HeatmapSquares(image_path, out_groups)
  output_image = hs.draw_all_groups(False, False)

  im = Image.fromarray(output_image)
  filename = os.path.basename(image_path)
  name = os.path.splitext(filename)[0]
  im.save(os.path.join(outdir,name + '_boundingbox.png'))


