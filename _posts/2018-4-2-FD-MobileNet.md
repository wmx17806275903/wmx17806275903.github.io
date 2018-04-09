---
layout:     post
title:      FD-MOBILENET （IMPROVED MOBILENET WITH A FAST DOWNSAMPLING STRATEGY）
subtitle:   使用快速下采样策略改善MobileNet
date:       2018-04-02
author:     WMX
header-img: img/qiu.jpg
catalog: true
tags:
    - 计算机视觉
    - 深度学习
    - FD-MobileNet
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

原始的MobileNet采用[缓慢的下采样]策略，当计算预算相对较小时，会导致严重的性能下降，例如10-140 MFLOPs。 在这种缓慢的下采样策略中，很多层具有较大的feature map，因此特征表示更加详细。 但是，网络中的信道数量有限，信息容量相对较小。 如果网络的宽度进一步缩小以适应非常有限的复杂性，信息容量将变得太小，网络的性能将崩溃。
在本文中，我们针对非常有限的计算资源（例如10到140个MFLOPs）提出了一种高效、准确的网络，称为快速下采样MobileNet（FD-MobileNet）。并非仅仅缩小网络的宽度以适应小的计算预算，我们通过在MobileNet框架中采用快速下采样策略来构建FD-MobileNet。在所提出的FD MobileNet中，我们在前12层内执行32倍下采样，这只是原始MobileNet中的一半。之后，为了更好的表示能力，采用了一系列深度可分的卷积。通过快速下采样策略，FD-MobileNet具有以下三个优点：（i）FD-MobileNet的计算代价随着feature map的空间维度减小而下降。（ii）在同样的复杂度下，FD-MobileNet比MobileNet有更多的通道。这极大的增加了FD-MobileNet的信息容量，这对于小型网络性能至关重要 （iii）FD-MobileNet继承了MobileNet的简单架构，并在工程实现中提供了快速的推理速度。

## FD-MobileNet的设计
FD-MobileNet由高效的深度可分离卷积组成，并采用快速下采样策略。

### 深度可分离卷积
FD-MobileNet利用深度可分离的卷积作为building block。 一个k×k的深度可分离卷积将一个k×k标准卷积分解为一个k×k深度卷积和一个逐点卷积（8-9倍减少FLOPs）。
实际上，深度可分离卷积可以实现与标准卷积相当的性能，同时在计算量有限的设备上可以提供很高的效率。 

### 快速下采样策略
现代CNN采用分层结构，同一阶段内各层的空间维度保持一致，下一阶段的空间维度通过下采样减少。鉴于有限的计算预算，紧凑型网络同时受到弱特征表示和受限制的信息容量的影响。不同的下采样策略在紧凑网络的详细特征表示和大信息容量之间提供了折衷。在一个缓慢的下采样策略中，下采样在网络后面的层中执行，因此更多的层具有较大的空间维度。相反，在快速下采样策略下在网络开始时执行下采样，这显着降低了计算成本。因此，在给定固定的计算预算的情况下，缓慢的下采样策略倾向于产生更多的细节特征，而快速下采样策略可以增加通道数量并允许编码更多的信息。

当计算预算非常小时，信息容量在网络性能中扮演更重要的角色。 通常，减少信道的数量以使紧凑的网络架构适应一定的复杂度。 在采用缓慢下采样方案的情况下，网络变得太窄而不能编码足够的信息，这导致严重的性能下降。例如，在12个MFLOPs的复杂度下，原始MobileNet架构在全局池之前的最后一层中只有128个通道，因此信息容量非常有限。

基于这种见解，我们建议在FD-MobileNet架构中采用快速下采样策略，并将特征提取过程推迟到最小分辨率。更快的下采样通过在网络开始处连续应用具有大步幅的深度可分离卷积来实现。在这里，我们不使用最大池，因为我们发现它没有获得性能改进，但引入了额外的计算。所提出的FD-MobileNet接受大小为224×224像素的图像，并且在前2个层内执行4x下采样，仅在12层内执行32x下采样，而原始MobileNet中执行相同下采样的层数分别为4和24。更具体地说，12层由1个标准卷积层，5个深度可分离卷积（每个卷积层具有深度卷积层和逐点卷积层）和1个深度卷积层组成。图1说明了FD MobileNet，MobileNet和ShuffleNet的下采样策略在140个MFLOP的计算预算下的比较。从图中可以看出，在feature map缩小到7×7之前，FD-MobileNet比其他架构浅得多。
![图1](https://img-blog.csdn.net/20180221132002687?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE5MTQ2ODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 剩余图层 
快速下采样策略的使用显着降低了最小空间维度（7×7）之前各层的计算成本。 在140个MFLOP的计算预算下，MobileNet在最大的4个分辨率上花费大约129个MFLOPs，而FD-MobileNet仅花费大约59个MFLOPs，如表1所示。因此，在所提出的架构中可以利用更多层和更多通道。 这里我们利用6个深度可分卷积来改善生成特征的表示能力。 前5个深度可分离卷积的输出通道是512，而最后一个是1024，这是MobileNet（0.5×MobileNet-224）的两倍。 通道数量的增加有助于提供更大的信息容量，这对于网络在极其有限的计算资源下的性能至关重要。
![表1](https://img-blog.csdn.net/20180221132410606?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE5MTQ2ODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 整体架构 
FD-MobileNet的总体结构如表1所示.FD-MobileNet采用24层的简单堆叠架构，包括1个标准卷积层，11个深度可分离卷积和1个完全连接层。然后，在每个卷积层之后应用批量归一化[23 ]和ReLU激活。 为了将FDMobileNet应用于不同的计算预算，我们引入了一个超参数α，称为宽度乘法器[18]，以统一调整FD-MobileNet的宽度。 我们用一个简单的符号“FDMobileNet α×”表示一个宽度乘数α的网络，表1中的网络表示为“FD-MobileNet 1×”
推断效率 目前的深度学习框架通过构建非循环计算图来完成对神经网络的推断。 对于移动或嵌入式设备，内存和缓存资源是有限的。 因此，复杂的计算图可能会导致频繁的内存/高速缓存切换，从而降低实际的推理速度。 FD-MobileNet继承了原始MobileNet的简单架构，并且计算图中只有一条信息路径。 这使FD-MobileNet对工程实施非常友好，并且在物理设备上非常有效。
## 实验
### 在ILSVRC 2012数据集上的结果
我们首先评估FD-MobileNet在ILSVRC 2012数据集上的有效性。ILSVRC 2012数据集由120万幅训练图像和50,000张验证图像组成。在实验中，网络在训练集上使用PyTorch进行训练，其中四个GPU用于90个epoch。[3]之后，batch大小设置为256，并使用0.9的动量。学习率从0.1开始，每30个衰减一个数量级。由于网络相对较小，因此按照[19]的建议使用4e-5的重量衰减。对于数据增强，我们采用稍微不太积极的多尺度增强方案，而不使用颜色抖动。在评估中，报告了验证集中的中心 - 作物top-1的准确率。首先将每个验证图像的边缘调整为256像素，然后使用中心224×224像素裁剪进行评估。表2展示了在三种计算预算下FD-MobileNet，MobileNet和ShuffleNet的top-1精度比较。
![表2](https://img-blog.csdn.net/20180221132553327?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE5MTQ2ODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
从表中可以看出，在不同的计算预算下，FD-MobileNet实现了比MobileNet更大的改进。据观察，在140个MFLOPs的复杂度下，FD-MobileNet以1.6％的幅度超过MobileNet，并且当计算预算分别为40和12个MFLOP时，FD-MobileNet的性能提高了5.6％和5.5％。值得注意的是，当计算预算非常小（例如40和12个MFOP）时，FD-MobileNet比MobileNet提供了显着的改进。我们将这些改进归因于FD-MobileNet中快速下采样策略的有效性。原始的MobileNet采用缓慢的下采样策略，因此更多的层具有相对较大的特征映射并且计算量更大。因此，MobileNet相对较窄以维持计算效率，这限制了信息容量。另一方面，FD-MobileNet利用更快的下采样策略，可以利用更多的信道并缓解信息容量的下降。例如，在12个MFLOPs下，MobileNet的最后一层只输出128个信道，而数字FD-MobileNet增加了一倍。信息容量的增加显着提高了FD-MobileNet的性能。
与ShuffleNet相比，FD-MobileNet可以达到可比较的或稍差的结果。 我们推测这些差异是由于ShuffleNet单元bypass connection结构的有效性所致。bypass connection结构在各种计算机视觉任务中被证明是强大的。 但是，在低功耗移动设备或嵌入式设备上，bypass connection结构会导致频繁的内存/高速缓存切换并损害实际的推理速度。 相反，FD-MobileNet的简单架构有助于高效利用内存和缓存。 
### 在Pascal voc2007数据集上的结果
我们还进一步在PASCAL VOC2007检测数据集上进行了大量实验[21]，以研究所提出的FD-MobileNet的泛化能力。 PASCAL VOC 2007数据集由大约10,000张图像组成，分为三个（训练/验证/测试）组。 在实验中，对探测器进行了VOC 2007训练集的训练，并对VOC 2007测试集的单模型结果进行了报告。我们采用Faster R-CNN检测管线，比较了FDMobileNet和MobileNet在600 ×分辨率下三个计算预算（140，40和12 MFLOPs）。 检测器训练15个时期，batch大小为1.学习率从1e-3开始，每5个时期除以10。 重量衰减设置为4e-5。 其他超参数设置遵循[5]中的原始faster R-CNN。 在测试过程中，300个proposal被发送到R-CNN子网，以产生最终预测。
结果的比较见表3.据观察，在不同的计算预算下，FD-MobileNet比MobileNet有显着的改进。在140个MFLOPs的计算预算下，FD MobileNet检测器在mAP上超过了MobileNet检测器1.6％的余量。当复杂度较低时，差距会扩大。当复杂度被限制在40和12个MFLOPs时，FD-MobileNet在移动网络上分别高出2.8％和3.6％。更具体地说，对于单类结果，FD-MobileNet在大多数类上表现比MobileNet好。从表3中可以看出，当计算预算较小时，FD-MobileNet比MobileNet提供了更显着的改进。例如，当计算预算为12个MFLOPs时，FD-MobileNet可以对诸如瓶子（5.3％），椅子（2.4％）和船（0.2％）等难以使用MobileNet的类实现一致的改进。这些改进已经证明FDMobileNet具有强大的转移学习泛化能力。
![表3](https://img-blog.csdn.net/20180221132725901?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE5MTQ2ODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 实际推断时间评估
为了研究在物理设备上的性能，我们进一步比较了基于ARM平台的FD-MobileNet，MobileNet和ShuffleNet的实际推断时间。 实验是在i.MX 6系列CPU（单核，800 MHz）上使用优化的NCNN框架进行的。
表4显示了三个紧凑网络在140,40和12个MFLOPs计算预算下的推断时间。与MobileNet相比，FD-MobileNet在三个计算预算下实现了比mobilenet大约1.1倍的加速。这些改进归因于FD-MobileNet快速下采样架构的有效性。与ShuffleNet相比，FD-MobileNet提供更快的推理速度。当计算预算为140和40个MFLOPs时，FD-MobileNet分别比ShuffleNet提高1.33倍和1.47倍。在12个MFLOPs的复杂度下，加速比提高了：FD MobileNet比ShuffleNet快1.82倍。值得注意的是，在140和40个MFLOPs下，ShuffleNet模型比FD-MobileNet模型有更少的MFLOPs，但它们要慢得多。这种放缓是由于ShuffleNet单元bypass connection结构的低效率造成的。在低功耗设备上，bypass connection结构会导致频繁的内存和高速缓存切换，从而降低实际的推理速度。相反，简单的堆栈结构允许FD-MobileNet更高效地利用内存和缓存，这有助于更快的实际推理速度。这些结果表明FDMobileNet在实际的移动或嵌入式应用中是有效的。
![表4](https://img-blog.csdn.net/20180221132914974?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE5MTQ2ODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 结论
在这项工作中，我们提出了快速下采样MobileNet（FDMobileNet），这是一个非常有效且精确的网络，用于非常有限的计算预算。 FD-MobileNet是通过在最先进的MobileNet框架中采用快速下采样策略而构建的。与原始MobileNet相比，快速下采样方案的使用允许使用更多通道，从而增加了网络的信息容量，并有助于 以显着的性能改进。 ILSVRC 2012分类数据集和PASCAL VOC 2007检测数据集上的实验表明，FD-MobileNet在不同的计算预算下一贯优于MobileNet。 对实际推断时间的评估表明，FD-MobileNet在同样复杂的基于ARM的设备上通过ShuffleNet实现了显着的加速。 对于未来的工作，我们计划在ShuffleNet等其他紧凑型网络中采用快速下采样策略，以获得更好的性能。
