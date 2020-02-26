---
layout:     post
title:      数据增强(Data Augmentation)
subtitle:   技巧
date:       2018-08-30
author:     crazyang
header-img: img/data.jpg
catalog: true
tags:
    - AI
---

>在训练过程中，网络优化是一方面，数据集的优化又是另一方面。数据集会存在各类样本不均匀的情况，也就是各类样本的数量不一样，有的甚至差别很大。为了让模型具有更强的鲁棒性，采用Data Augmentation是一个不错的选择。


# 常用的方法

- Color Jittering：对颜色的数据增强：图像亮度、饱和度、对比度变化（此处对色彩抖动的理解不知是否得当）
- PCA Jittering：首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值。参见[论文](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Random Scale：尺度变换
- Random Crop：采用随机图像差值方式，对图像进行裁剪、缩放；包括Scale Jittering方法（VGG及ResNet模型使用）或者尺度和长宽比增强变换
- Horizontal/Vertical Flip：水平/垂直翻转
- Shift：平移变换
- Rotation/Reflection：旋转/仿射变换
- Noise：高斯噪声、模糊处理
- Label shuffle：类别不平衡数据的增广，参见海康威视ILSVRC2016的report

----------

# ImageDataGenerator()函数
这是Keras提供的一个自动增强的函数
```python
ImageDataGenerator(
	featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())
```
参数解释，所有参数不一定都需要包括，可以不写。
```
featurewise_center：布尔值，使输入数据集去中心化（均值为0）
samplewise_center：布尔值，使输入数据的每个样本均值为0
featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化
samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差
zca_whitening：布尔值，对输入数据施加ZCA白化
rotation_range：整数，数据提升时图片随机转动的角度
width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
channel_shift_range：浮点数，随机通道偏移的幅度
fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
horizontal_flip：布尔值，进行随机水平翻转
vertical_flip：布尔值，进行随机竖直翻转
rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
dim_ordering：‘tf’和‘th’之一，规定数据的维度顺序。‘tf’模式下数据的形状为samples, width, height, channels，‘th’下形状为(samples, channels, width, height).该参数的默认值是Keras配置文件~/.keras/keras.json的image_dim_ordering值,如果你从未设置过的话,就是'th'
```
有了这个函数，那怎么具体对一张图片或者批量化处理
```
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
	rotation_range=0.2,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')
	
img = load_img('lena.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x,
	batch_size=1,
	save_to_dir='data/preview',   #保存在这个文件夹下
	save_prefix='lena',
	save_format='jpg'):
    i += 1
    if i > 20:  #生成20张图
        break
```
其中flow函数的参数解释
```
X：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
batch_size：整数，默认32
shuffle：布尔值，是否随机打乱数据，默认为True
save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
save_format：”png”或”jpeg”之一，指定保存图片的数据格式,默认”jpeg”
yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
seed: 整数,随机数种子
```
**以上所有详情可以参考[keras官方文档](https://keras.io/preprocessing/image/)**
**如果你对每一步的效果不够清楚，请参考[参数演示](https://zhuanlan.zhihu.com/p/30197320)**

----------


# PCA Jittering方法
PCA是主成分分析，它是为了减少数据集的维数，同时保持数据集中的对方差贡献最大的特征，保留低阶主成分，忽略高阶主成分。**实施过程是对协方差矩阵进行特征分解，得到特征值和特征向量。** 关于PCA，可以去查看链接。

根据AlexNet论文，内容描述为：
>添加多个找到的主成分，其幅度与相应的特征值成正比，并要乘以一个零均值，标准差为0.1的高斯变量，对每一个RGB值 I<sub>xy</sub>=[I<sub>xy</sub><sup>R</sup> , I<sub>xy</sub><sup>G</sup> , I<sub>xy</sub><sup>B</sup> ] 要添加[p<sub>1</sub> , p<sub>2</sub> , p<sub>3</sub>] [α<sub>1</sub>λα<sub>1</sub> ， α<sub>2</sub>λα<sub>2</sub> ， α<sub>3</sub>λα<sub>3</sub>] ，p和λ是协方差矩阵的特征向量和特征值，α是高斯变量，该高斯变量的标准差参数也是唯一引入的外部变量

复现代码如下（使用方法在后面）：
```python
import numpy as np
import os
from PIL import Image, ImageOps
import random
from scipy import misc
import imageio
 
def PCA_Jittering(path):
    img_list = os.listdir(path)
    img_num = len(img_list)
    
    for i in range(img_num):
        img_path = os.path.join(path, img_list[i])
        img = Image.open(img_path)    
        
        img = np.asanyarray(img, dtype = 'float32')

        img = img / 255.0
        img_size = img.size // 3    #转换为单通道
        img1 = img.reshape(img_size, 3)

        img1 = np.transpose(img1)   #转置
        img_cov = np.cov([img1[0], img1[1], img1[2]])    #协方差矩阵
        lamda, p = np.linalg.eig(img_cov)     #得到上述协方差矩阵的特征向量和特征值
        
        #p是协方差矩阵的特征向量
        p = np.transpose(p)    #转置回去

        #生成高斯随机数********可以修改
        alpha1 = random.gauss(0,3)
        alpha2 = random.gauss(0,3)
        alpha3 = random.gauss(0,3)
        
        #lamda是协方差矩阵的特征值
        v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))     #转置

        #得到主成分
        add_num = np.dot(p,v)
        
        #在原图像的基础上加上主成分
        img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])

        #现在是BGR，要转成RBG再进行保存
        img2 = np.swapaxes(img2,0,2)
        img2 = np.swapaxes(img2,0,1)
        save_name = 'pre'+str(i)+'.png'
        save_path = os.path.join(path, save_name)
        misc.imsave(save_path,img2)
        
        #plt.imshow(img2)
        #plt.show()

PCA_Jittering('testpic')
```

使用方法是：主函数PCA_Jitterring括号中的参数是当前路径下的一个文件夹，它会自动加载文件夹中的图片。调整该方法的唯一参数（高斯函数的标准差）是在标记了生成高斯随机数那里，改变参数会得到很不同的效果。

----------
# Label Shuffling
由于场景数据集不均匀的类别分布，给模型训练带来了困难。海康威视提出了Label Shuffling的类别平衡策略。在Class-Aware Sampling方法中，定义了2种列表，一是类别列表，一是每个类别的图像列表，对于80类的分类问题来说，就需要事先定义80个列表，很不方便。对此进行了改进，只需要原始的图像列表就可以完成同样的均匀采样任务。

步骤如下：

> 首先对原始的图像列表，按照标签顺序进行排序；然后计算每个类别的样本数量，并得到样本最多的那个类别的样本数。根据这个最多的样本数，对每类随机都产生一个随机排列的列表；然后用每个类别的列表中的数对各自类别的样本数求余，得到一个索引值，从该类的图像中提取图像，生成该类的图像随机列表；然后把所有类别的随机列表连在一起，做个Random Shuffling，得到最后的图像列表，用这个列表进行训练。每个列表，到达最后一张图像的时候，然后再重新做一遍这些步骤，得到一个新的列表，接着训练。Label Shuffling方法的优点在于，只需要原始图像列表，所有操作都是在内存中在线完成，非常易于实现。

用一幅图片可以生动地展示这个过程
![](https://i.imgur.com/QzH9wMp.png)

实现的代码如下：

```python
import random

category=80

f=open('scene_train_20170904.txt')  #按照label从小到大排序
#f=open('scene_validation_20170908.txt')
dicts={}
for line in f:
    line=line.strip('\n')
    image=line.split()[0]
    label=int(line.split()[-1])
    dicts[image]=label
dicts=sorted(dicts.items(),key=lambda item:item[1])
f.close()

counts={}   #统计每一类label的数目
new_dicts=[]
for i in range(category):
    counts[i]=0
for line in dicts:
    line=list(line)
    line.append(counts[line[1]])
    #print line
    counts[line[1]]+=1
    new_dicts.append(line)
#print counts

#for line in new_dicts:
#    print line

tab=[]  #把原列表按照每一类分成各个block并形成新列表
origin_index=0
for i in range(category):
    block = []
    for j in range(counts[i]):
        block.append(new_dicts[origin_index])
        origin_index+=1
    #print block
    tab.append(block)
#print tab

nums=[] #找到数目最多的label类别
for key in counts:
    nums.append(counts[key])
nums.sort(reverse=True)
#print nums

lists=[]    #形成随机label序列
for i in range(nums[0]):
    lists.append(i)
#print lists
all_index=[]
for i in range(category):
    random.shuffle(lists)
    #print lists
    lists_res=[j%counts[i] for j in lists]
    all_index.append(lists_res)
    #print lists_res
#print all_index

#f=open('train_shuffle_labels.txt','w') #按照随机序列提取图像生成最后的标签
#f=open('val_shuffle_labels.txt','w')
f=open('train_shuffle_labels.lst','w')
#f=open('val_shuffle_labels.lst','w')
shuffle_labels=[]
index=0
for line in all_index:
    for i in line:
        shuffle_labels.append(tab[index][i])
    index+=1
#print shuffle_labels
random.shuffle(shuffle_labels)
id=0
for line in shuffle_labels:
    #print line
    #f.write(line[0]+' '+str(line[1]))
    f.write(str(id)+'\t'+str(line[1])+'\t'+line[0])
    f.write('\n')
    id+=1
f.close()
```

----------
另外我觉得单纯地复制粘贴会不会有用呢
