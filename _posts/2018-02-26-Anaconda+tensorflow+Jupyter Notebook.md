---
layout:     post
title:      如何在Anaconda下使用jupyter notebook
subtitle:   在jupyter notebook中使用tensorflow
date:       2018-02-26
author:     WMX
header-img: img/tech-eye.jpg
catalog: true
tags:
    - Anaconda
    - tensorflow
    - jupyter notebook
---


最初接触到这个在线编程web应用程序是毕设学习Python时，视频中那位老师在使用这个工具，因为已经安装了pycharm编辑器，所以当时没有想去了解。
而今，刚好电脑装了Anaconda，直接在web端编程，避免浏览器，编辑器的切换，就在Anaconda环境下搭建了一下jupyter notebook，方便直接调用。

#什么是jupyter notebook

（Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本，支持运行 40 多种编程语言。
Jupyter Notebook 的本质是一个 Web 应用程序，便于创建和共享文学化程序文档，支持实时代码，数学方程，可视化和 markdown。 用途包括：数据清理和转换，数值模拟，统计建模，机器学习等等 ）  -----摘自 百度百科

所以可以看到Jupyter Notebook功能多样，是一款比较强大的应用程序。

#在Anaconda中配置tensorflow

1.打开终端

![](http://bmob-cdn-16714.b0.upaiyun.com/2018/02/26/785c41784031003f80b415ea20147598.png)

2.创建一个叫做tensorflow的conda环境并激活

  conda create -n tensorflow python=3.5（这里注意，tensorflow目前只支持Python3.5）
  
  activate tensorflow （ 激活，激活之后环境前面就是以这种形式显示的：(tensorflow)C:>  ）
  
3.CPU版本的tensorflow，如果需要安装GPU版本的自行搜索哦。输入以下命令：

  pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl

4.至此，我们已经安装好了TensorFlow。接下来可以测试验证下是否可以使用。在命令行输入：Python，进入Python编程环境。
然后输入：

>>> import tensorflow as tf  
>>> hello = tf.constant('Hello, TensorFlow!')  
>>> sess = tf.Session()  
>>> print(sess.run(hello)) 
 
正确的话，会输出：
Hello, TensorFlow!  

#在Jupyter中使用TensorFlow

1.首先，进入tensorFlow环境，安装ipython和jupyter。命令如下：

 conda install ipython  
 
 conda install jupyter 

2.然后，进去Jupyter，直接输入：

 jupyter notebook （等待一下就会弹出浏览器，进入Jupyter） 

3.最后，可以进行测试导入tensorflow，验证过程同上
 import tensorflow as tf  
 。。。
可以成功运行啦！
