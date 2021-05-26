# 面部表情识别

## 简介

面部表情识别系统。用于寻找图片中的面部并识别出表情所属的类别。使用卷积神经网络构建整个系统，并在FER2013、JAFFE和CK+三个表情识别数据集上进行模型评估。

## 环境部署

基于Python3和Keras2（TensorFlow后端）

```
pip install -r requirements.txt
```

## 数据准备

使用FER2013、JAFFE和CK+数据集进行训练，下载数据集之后将其解压存放在dataset目录之下

## 网络设计

使用卷积神经网络，参考A Compact Deep Learning Model for Robust Facial Expression Recognition论文实现

![](.\assets\CNN.png)

