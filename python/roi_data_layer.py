import caffe
import cv2
import random
import numpy as np


class RoiDataLayer(caffe.Layer):
  
  """
  This is a simple datalayer for training SpindleNet.
  """

  def setup(self, bottom, top):
    params = eval(self.param_str)
    self.source = params['source']
    self.root_folder = params['root_folder']
    self.batch_size = params['batch_size']
    self.new_height = params['new_height']
    self.new_width = params['new_width']
    self.shuffle = params['shuffle']
    self.mirror = params['mirror']
    self.mean_value = params['mean_value']
    self.region_num = params['region_num']
    self.region_scale = params['region_scale']

    assert self.batch_size > 0
    assert self.new_height > 0 and self.new_width > 0
    assert type(self.shuffle) == bool
    assert type(self.mirror) == bool
    assert len(self.mean_value) == 3
    assert type(self.region_scale) == bool
    
    self.spindle_datas = []

    with open(self.source) as f:
      while True:
        line = f.readline()
        if not line:
          break
        image_index = int(line.split(' ')[1])
        if image_index % 10000 == 0:
          print "processing " + str(image_index)
        image_path = self.root_folder + f.readline().strip()
        label = int(f.readline())
        regions = []
        for i in xrange(self.region_num):
          line = f.readline()
          tmp = []
          tmp.append(float(line.split(' ')[0]))
          tmp.append(float(line.split(' ')[1]))
          tmp.append(float(line.split(' ')[2]))
          tmp.append(float(line.split(' ')[3]))
          regions.append(tmp)
        self.spindle_datas.append(SpindleData(image_path, label, regions))

    assert len(self.spindle_datas) > 0

    if self.shuffle == True:
      print "Shuffling data"
      random.shuffle(self.spindle_datas)

    print "A total of " + str(len(self.spindle_datas)) + " images."

    self.cur = 0
    cv_img = cv2.imread(self.spindle_datas[self.cur].image_path)
    assert cv_img != None
    
    print "output data size: " + str(self.batch_size) + "," + str(cv_img.shape[2]) + "," + str(self.new_height) + "," + str(self.new_width)

    top[0].reshape(self.batch_size, cv_img.shape[2], self.new_height, self.new_width)
    top[1].reshape(self.batch_size)
    for i in xrange(self.region_num):
      top[2 + i].reshape(self.batch_size, 5)


  def forward(self, bottom, top):
    for i in xrange(self.batch_size):
      tmp = self.spindle_datas[self.cur]
      cv_img = cv2.imread(tmp.image_path)
      assert cv_img != None
      assert cv_img.shape[0] > 1
      assert cv_img.shape[1] > 1
      assert cv_img.shape[2] == 3
      
      x_scale = 1.0
      y_scale = 1.0
      if self.region_scale == True:
        x_scale = (self.new_width - 1) * 1.0 / (cv_img.shape[1] - 1)
        y_scale = (self.new_height - 1) * 1.0 / (cv_img.shape[0] - 1)

      cv_img = cv2.resize(cv_img, (self.new_width, self.new_height))
      flip = 1
      if self.mirror == True:
        flip = np.random.choice(2) * 2 - 1
      cv_img = cv_img[:,::flip,:]
      cv_img = np.array(cv_img, dtype=np.float32)
      cv_img[:,:,0] -= self.mean_value[0]
      cv_img[:,:,1] -= self.mean_value[1]
      cv_img[:,:,2] -= self.mean_value[2]

      cv_img = cv_img.transpose((2,0,1))

      self.cur += 1
      if self.cur >= len(self.spindle_datas):
        if self.shuffle == True:
          print "Shuffling data"
          random.shuffle(self.spindle_datas)
        self.cur = 0
      
      top[0].data[i,...] = cv_img
      top[1].data[i,...] = tmp.label
      for j in xrange(self.region_num):
        regions = np.zeros(5)
        regions[0] = i
        regions[2] = tmp.regions[j][1] * y_scale;
        regions[4] = tmp.regions[j][3] * y_scale;
        if flip == -1:
          regions[1] = self.new_width - 1 - tmp.regions[j][2] * x_scale;
          regions[3] = self.new_width - 1 - tmp.regions[j][0] * x_scale;
        else:
          regions[1] = tmp.regions[j][0] * x_scale;
          regions[3] = tmp.regions[j][2] * x_scale;
        top[2 + j].data[i,...] = regions


  def reshape(self, bottom, top):
    """
    There is no need to reshape the data, since the input is of fixed size (rows and colums)
    """
    pass


  def backward(self, top, propagate_down, bottom):
    """
    This layer does not back propagate
    """
    pass


class SpindleData(object):
  
  def __init__(self, image_path, label, regions):
    self.image_path = image_path
    self.label = label
    self.regions = regions

