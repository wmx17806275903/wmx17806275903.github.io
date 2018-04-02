---
layout:     post
title:      FD-MOBILENET IMPROVED MOBILENET WITH A FAST DOWNSAMPLING STRATEGY
subtitle:   FD-MOBILENET 使用下采样策略改善MobileNet
date:       2018-4-2
author:     WMX
header-img: img/tech-eye.jpg
catalog: true
tags:
    - 计算机视觉
    - 深度学习
    - FD-MobileNet
    - 下采样策略
    - 深度可分离卷积
---

# 前言
GitHub原文 [点这里 ->> ](https://github.com/wmx17806275903/wmx17806275903.github.io/new/master/_posts)
如果对您有帮助，请给我小星星（Star一下呗）。

这篇文章发表于CVPR2018[FD-MOBILENET: IMPROVED MOBILENET WITH A FAST DOWNSAMPLING STRATEGY](https://arxiv.org/pdf/1802.03750.pdf)

## 参考文献 (轻量级网络论文链接，备用)

【18】[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/pdf/1704.04861.pdf)

【19】[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
Devices](https://arxiv.org/pdf/1707.01083.pdf)

## 介绍
该文的主要想法是对MobileNet框架应用快速下采样策略。在FD-MobileNet中，我们在12层内执行32倍下采样，只有原始MobileNet中层数的一半。
这种设计有三个优点：
（i）显着降低了计算成本。 
（ii）增加信息容量并实现显着的性能改进;
（iii）工程友好并提供快速的实际推理速度。在ILSVRC 2012和PASCAL VOC 2007数据集上的实验表明，FD-MobileNet一直超越MobileNet，并在不同的计算预算
下取得与ShuffleNet相似的结果，例如，在ILSVRC 2012上排名第一的精度上超过MobileNet 5.5％，VOC2007在12个MFLOP的复杂度下超过3.6％ MAP。
在基于ARM的设备上，FD-MobileNet在相同的复杂度下实现了比MobileNet高1.11倍的推理加速比和比ShuffleNet高1.82倍。

## FD-MobileNet的设计
FD-MobileNet由高效的深度可分离卷积组成，并采用快速下采样策略。
FD-MobileNet利用深度可分离的卷积作为building block。 一个k×k的深度可分离卷积将一个k×k标准卷积分解为一个k×k深度卷积和一个逐点卷积（8-9倍减少FLOPs）。
实际上，深度可分离卷积可以实现与标准卷积相当的性能，同时在计算量有限的设备上可以提供很高的效率。 
