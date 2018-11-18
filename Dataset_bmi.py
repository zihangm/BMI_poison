import numpy as np
import os
import scipy.misc
import random
import pickle 
import pandas as pd
import copy


f = open('./BMI_data/Data/table_validation_final.pkl','rb')
df = pd.read_pickle(f)
global_ave_num = 1000
trainsize = 7000

class reader:
  def __init__(self):
    self.bmi_dic = {}
    self.dataset = []
    self.trainset = []
    self.testset = []
    self.imgdir = './BMI_data/Data/data_UW/'
    self.batch_pointer = 0
    self.test_pointer = 0
    self.global_mean =  self.get_global_mean()
    self.read_img()

  def get_global_mean(self):
    img_name_list = os.listdir(self.imgdir)
    img_list = []
    for i in range(global_ave_num): # for now, just use mean of first 1000 
      img = scipy.misc.imresize(scipy.misc.imread(self.imgdir+img_name_list[i]), [256,256])
      img_list.append(img)
    global_mean = np.mean(np.array(img_list),axis=(0,1,2))
    return global_mean

  def read_img(self):
    for img_name in os.listdir(self.imgdir):
      img = np.float32(scipy.misc.imresize(scipy.misc.imread(self.imgdir+img_name), [256,256]))
      img[:,:,0] = img[:,:,0] - self.global_mean[0]
      img[:,:,2] = img[:,:,1] - self.global_mean[1]
      img[:,:,2] = img[:,:,2] - self.global_mean[2]
      tmp0 = copy.deepcopy(img[:,:,0])
      tmp2 = copy.deepcopy(img[:,:,2])
      img[:,:,0] = tmp2
      img[:,:,2] = tmp0
      self.dataset.append([img, img_name])
    #random.shuffle(self.dataset)
    self.trainset = self.dataset[0:trainsize]
    self.testset = self.dataset[trainsize:]
    for i in range(9045):
      self.bmi_dic[str(df.DCNumber[i])+'.jpg'] = df.bmi[i]
    return 
        
  def find_max_num(self,l):
    max_value = max(l)
    count = 0
    for i in range(3):
      if l[i] == max_value:
        count = count + 1
    return count
  
  def rand_crop(self, img):
    flip = random.randint(0,1)
    if flip==1:
      img[:,:,0] = np.fliplr(img[:,:,0])
      img[:,:,1] = np.fliplr(img[:,:,1])
      img[:,:,2] = np.fliplr(img[:,:,2])
    rand_x = random.randint(0,23)
    rand_y = random.randint(0,23)
    return img[rand_x:rand_x+227, rand_y:rand_y+227, :]

  def crop(self, img, k):
    if k>=5:
      k = k-5
      img[:,:,0] = np.fliplr(img[:,:,0])
      img[:,:,1] = np.fliplr(img[:,:,1])
      img[:,:,2] = np.fliplr(img[:,:,2])
    if k==0:
      return img[8:235,8:235,:]
    if k==1:
      return img[0:0+227,0:0+227,:]
    if k==2:
      return img[0:0+227,23:23+227,:]
    if k==3:
      return img[23:23+227,0:0+227,:]
    if k==4:
      return img[23:23+227,23:23+227,:]


  def next_batch(self, batch_size, num_nodes):
    next_batch = [[], []]
    epoch_end = 0
    for i in range(num_nodes):
      data_batch = self.trainset[self.batch_pointer:(self.batch_pointer+batch_size)]
      self.batch_pointer = self.batch_pointer + batch_size
      img_batch = [self.rand_crop(data_batch[j][0]) for j in range(batch_size)]
      label_batch = self.get_label(data_batch)
      next_batch[0].append(np.array(img_batch))
      next_batch[1].append(np.array(label_batch))
      if self.batch_pointer >= trainsize - batch_size:
        self.batch_pointer = 0
        epoch_end=1
        random.shuffle(self.trainset)
    next_batch[1] = np.array(next_batch[1])
    return next_batch, epoch_end
  
  def get_label(self, data_batch):
    batch_label = [self.bmi_dic[data_batch[i][1]] for i in range(len(data_batch))]
    return batch_label


  def test_batch(self, batch_size, num_nodes):
   next_batch = [[],[]]
   self.test_pointer = 0
   for i in range(num_nodes):
     data_batch = self.testset[self.test_pointer:(self.test_pointer+batch_size)]
     img_batch = [self.crop(data_batch[j][0], 0) for j in range(batch_size)]
     label_batch = self.get_label(data_batch)
     next_batch[0].append(np.array(img_batch))
     next_batch[1].append(np.array(label_batch))
     self.test_pointer = self.test_pointer + batch_size
   next_batch[1] = np.array(next_batch[1])
   return next_batch

