---
slug: VGG-Very Deep Convolutional Networks for Large-Scale Image Recognition
title: VGG-Very Deep Convolutional Networks for Large-Scale Image Recognition
authors: [YunhaoLi,]
tags: [Deeplearning, Image Classification]
---
# VGGNet

<!-- Authors: Simonyan, Karen
Zisserman, Andrew
Created time: August 2, 2022 3:54 PM
DOI: https://doi.org/10.48550/arXiv.1409.1556
File Path: /Users/joyce/Zotero/storage/8VFW8XDF/Simonyan 和 Zisserman - 2015 - Very Deep Convolutional Networks for Large-Scale I.pdf
Finish time: 2022/09/01
Full Citation: Simonyan, Karen, 和Andrew Zisserman. 《Very Deep Convolutional Networks for Large-Scale Image Recognition》. arXiv, 2015年4月10日. https://doi.org/10.48550/arXiv.1409.1556.
Future: 耗费更多计算资源，并且使用了更多的参数，导致更多的内存占用
In-Text Citation: (Simonyan & Zisserman, 2015)
Item Type: 预印本
Meaning: 参数总数基本不变的情况下，CNN随着层数的增加，其效果的变化
Theoretical/Conceptual Framework: CNN
Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
Year: 2015
Zotero URI: http://zotero.org/users/9806271/items/8KS5QRJ5
关键词: AI, Classic
期刊杂志: ICLR2015
管理用｜文献管理状态: 已完成 -->

# 1. 第一遍 标题 摘要 结论

## 1⃣️ 标题

Very Deep Convolutional Networks For Large-Scale Image Recognition

- `Deep Convolutional`：`卷积神经网络`工作原理是什么? 同时作者为什么要使用`深度`的卷积神经网络。跟AlexNet的区别
- `Large-Scale Image Recognition` ：大规模图像分类

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled.png)

## 2⃣️ 摘要

- 干了什么？
    
    实验了在大规模图像数据集分类上卷积神经网络的深度对精度的影响
    
- 如何做的？
    
    增加一个使用(3,3)作为convolution filters的卷积神经网络的深度，在当时的条件下，16-19权重层能达到很好的效果
    
- 效果如何？
    
    在ImageNet Challenge 2014挑战赛中，分别在localisation和classification中拿到第一第二名
    
- 应用范围？
    
    在其他数据集上也能得到很好的效果，本论文提供了两个最好效果的ConvNet models给后人研究
    
    ![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%201.png)
    

## 3⃣️ 结论

与摘要前后呼应，再次强调卷积神经网络的深度对分类精确度的重要性，再次自夸自己的模型效果有多好，有多适应各种数据集

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%202.png)

# 2. 第二遍 全部过一遍

## 一、基本信息

标题：Very deep convolutional networks for large-scale image recognition
时间：2014
出版源：arXiv
论文领域：CNN、深度学习
引用格式：Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

## 二、研究背景

在这项工作中，我们研究了在大规模图像识别中卷积网络的深度对其准确性的影响。

我们的主要贡献是对使用带有非常小(3 * 3)卷积滤波器的结构来增加深度的网络进行了深入评估，结果显示，通过将深度推到16-19个权重层，可以显著改善以前的配置。

CNN取得成功，原因：

- 大型训练集，如ImageNet
- 高性能计算GPU以及分布式计算
- ImageNet Large-ScaleVisual Recognition Challenge

许多人对AlexNet改进：

- 更小的感受窗口尺寸和更小的第一卷积层步长； ILSVRC-2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014)
- 多尺度，在整个图像和多个尺度上对网络进行密集地训练和测试；Sermanet et al., 2014; Howard, 2014
- 本文侧重对卷积深度的改进，使用很小的核3 * 3

## 三、创新点——深度

输入：224 * 224

预处理：每个像素RGB 减去 训练集RGB均值

3个3 * 3 与单个 7 * 7：

- 3和卷积带来的识别能力更强
- 参数更少
- 使用1 * 1卷积，增加决策函数而不影响卷积层接收域

小尺寸卷积核：
GoogleNet也使用了更深的网络（22层），更小的卷积核3 * 3 ，同样使用 1 *1卷积，其更复杂，在第一层降低了特征图的空间分辨率，以减少计算量。单网络分类准确度方面，本文由于GoogleNet。

1）使用步长为1的1*1卷积，以增加网络的非线性；

2）3*3卷积使用same padiing，不降低输出特征图大小；

3）所有的隐层都配置了Relu。

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%203.png)

conv3-64 表示 3 * 3 卷积核大小，数目64

上图从左到右：

- 8个卷积层 + 3个全连接层 到 16个卷积层 + 3个全连接层
- 卷积层的宽度（通道数）更小，从64到512，每次池化翻倍
- 虽然深度变多，但是没有比大卷积核的网络更多，可以看对最多144M

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%204.png)

### **文章中的一些讨论/结论**

**通过使用三个3×3卷积层的堆叠来替换单个7×7层，获得了什么？**

- 感受野不变。在整个网络使用非常小的3×3感受野，与输入的每个像素（步长为1）进行卷积。两个3×3卷积层堆叠（没有空间池化）有5×5的有效感受野；三个这样的层具有7×7的有效感受野。
- 结合了三个非线性修正层、增加了非线性，而不是单一的，这使得决策函数更具判别性；
- 减少参数的数量。假设三层3×3卷积堆叠的输入和输出有C个通道，堆叠卷积层的参数为3*(3^2C^2)=27C^2个权重；
- 起到正则化作用。单个7×7卷积层将需要7^2C^2=49C^2个参数，即参数多81％。这可以看作是对7×7卷积滤波器进行正则化，迫使它们通过3×3滤波器（在它们之间注入非线性）进行分解。

5x5卷积看做一个小的全连接网络在5x5区域滑动，我们可以先用一个3x3的卷积滤波器卷积，然后再用一个全连接层连接这个3x3卷积输出，这个全连接层我们也可以看做一个3x3卷积层。这样我们就可以用两个3x3卷积级联（叠加）起来代替一个 5x5卷积。

具体如下图所示：

### **1*1卷积核的使用**

结合1×1卷积层（配置C，表1）是增加决策函数非线性而不影响卷积层感受野的一种方式

### **开始输入层的小卷积核的使用**

ILSVRC-2013比赛时，在第一卷积层中使用相对较大的感受野（ResNet也沿用了这种结构）。本文整个网络都使用3*3Conv，15年之后第一层大卷积就很少见了。

1）训练参数：

- 批量大小设为256，动量为0.9，L2惩罚乘子设定为5⋅10−4，前两个全连接层执行丢弃正则化（丢弃率设定为0.5）
- 学习率初始设定为10−2，当验证集准确率停止改善时，减少10倍。学习率总共降低3次

2）参数比Alex多，但收敛更早，作者推测：

a）由更大的深度和更小的卷积滤波器尺寸引起的隐式正则化；

b）某些层的预初始化

3）数据扩充，与AlexNet一致。水平翻转、随机剪裁和随机RGB颜色偏移。

## 四、训练

### 4.1、训练技巧——多尺度训练方式

相比AlexNet的随机剪裁、随机颜色增广外，开启了进行了多尺度训练图像中对输入。这样的对比试验，在19年，mmdetection发表的论文中，仍进行了更深层次的实验对比。

**训练图像大小**：S为训练尺度，原图按最小边等比例缩放（AlexNet中使用的方式）。之后采用下面两种方式训练：

第一种是修正对应单尺度训练的S。用两个固定尺度训练的模型：S=256和S=384。给定ConvNet配置，先使用S=256来训练网络，之后用S=256预训练的权重来进行初始化，加速S=384网络的训练，学习率使用10−3。

第二种是多尺度训练。其中每个训练图像通过从一定范围[Smin，Smax]（使用Smin=256和Smax=512）随机采样S来单独进行归一化。由于图像中的目标可能具有不同的大小，因此在训练期间考虑到这一点是有益的。这也可以看作是通过尺度抖动进行训练集增强，其中单个模型被训练在一定尺度范围内识别对象。为了速度的原因，我们通过对具有相同配置的单尺度模型的所有层进行微调，训练了多尺度模型，并用固定的S=384进行预训练。

### 4.2、训练技巧-加速收敛

参数相比AlexNet更多，网络的深度更大，但网络需要更小的epoch就可以收敛，原因：

1、更大的深度和更小的卷积滤波器尺寸引起的隐式正则化。统统使用了3*3的卷积核，相比AlexNet的5*5,7*7,11*11卷积小了很多；

2、某些层的初始化。作者首先训练了网络A，因为A比较小，所以更容易收敛。训练好A后，得到model去finetune 网络C，可以一次类推进而得到网络E。使用这种训练方法，显然可以加快收敛。

### 4.3、常规设置

**目前仍在使用的一些设置：**

- 批量大小设为256，动量为0.9。
- 训练通过权重衰减（L2惩罚乘子设定为5⋅10−4）进行正则化，前两个全连接层执行丢弃正则化（丢弃率设定为0.5）
- 学习率初始设定为10−2，然后当验证集准确率停止改善时，减少10倍。学习率总共降低3次
- 数据扩充，与AlexNet一致。水平翻转、随机剪裁和随机RGB颜色偏移。

## 五、实验结果

### 单尺度验证

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%205.png)

- 使用局部响应归一化（A-LRN网络）在没有任何归一化层的情况下，对模型A没有改善。因此，我们在较深的架构（B-E）中不采用归一化
- 观察到分类误差随着ConvNet深度的增加而减小：从A中的11层到E中的19层。值得注意的是，尽管深度相同，配置C（包含三个1×1卷积层）比在整个网络层中使用3×3卷积的配置D更差。这表明，虽然额外的非线性确实有帮助（C优于B），但也可以通过使用具有非线性感受野（D比C好）的卷积滤波器来捕获空间上下文
- 当深度达到19层时，我们架构的错误率饱和，但更深的模型可能有益于较大的数据集。我们还将网络B与具有5×5卷积层的浅层网络进行了比较，浅层网络可以通过用单个5×5卷积层替换B中每对3×3卷积层得到（其具有相同的感受野如第2.3节所述）。测量的浅层网络top-1错误率比网络B的top-1错误率（在中心裁剪图像上）高7％，这证实了具有小滤波器的深层网络优于具有较大滤波器的浅层网络
- 训练时的尺度抖动（S∈[256;512]）得到了与固定最小边（S=256或S=384）的图像训练相比更好的结果，即使在测试时使用单尺度。这证实了通过尺度抖动进行的训练集增强确实有助于捕获多尺度图像统计

### 多尺度验证

考虑到训练和测试尺度之间的巨大差异会导致性能下降，用固定S训练的模型在三个测试图像尺度上进行了评估，接近于训练一次：Q=S−32,S,S+32。同时，训练时的尺度抖动允许网络在测试时应用于更广的尺度范围，所以用变量S∈[Smin;Smax]训练的模型在更大的尺寸范围Q = {S_{min}, 0.5(S_{min} + S_{max}), S_{max}上进行评估。

表4中给出的结果表明，测试时的尺度抖动导致了更好的性能（与在单一尺度上相同模型的评估相比，如表3所示）。

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%206.png)

对于固定尺度S:Q = {S − 32, S, S + 32}.

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%207.png)

### **多裁剪图像评估**

在表5中，我们将稠密ConvNet评估与多裁剪图像评估进行比较（细节参见第3.2节）。我们还通过平均其soft-max输出来评估两种评估技术的互补性。可以看出，使用多裁剪图像表现比密集评估略好，而且这两种方法确实是互补的，因为它们的组合优于其中的每一种。如上所述，我们假设这是由于卷积边界条件的不同处理。

表5：ConvNet评估技术比较。在所有的实验中训练尺度S从[256；512]采样，三个测试适度Q考虑：{256, 384, 512}。

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%208.png)

### **卷积网络融合**

结果如表6所示。在ILSVRC提交的时候，我们只训练了单规模网络，以及一个多尺度模型D（仅在全连接层进行微调而不是所有层）。由此产生的7个网络组合具有7.3％的ILSVRC测试误差。在提交之后，我们考虑了只有两个表现最好的多尺度模型（配置D和E）的组合，它使用密集评估将测试误差降低到7.0％，使用密集评估和多裁剪图像评估将测试误差降低到6.8％。作为参考，我们表现最佳的单模型达到7.1％的误差（模型E，表5）。

表6：多个卷积网络融合结果：

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%209.png)

![Untitled](VGGNet%20c93c17c7aa664b7e8b1dd71bbc7b82b5/Untitled%2010.png)

## **VGG优缺点**

**VGG优点**

- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。

**VGG缺点**

- VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG可是有3个全连接层啊！

PS：有的文章称：发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量。

注：很多pretrained的方法就是使用VGG的model（主要是16和19），VGG相对其他的方法，参数空间很大，最终的model有500多m，AlexNet只有200m，GoogLeNet更少，所以train一个vgg模型通常要花费更长的时间，所幸有公开的pretrained model让我们很方便的使用。

## ***怎么增加模型判别性？***

论文中提到使 3 个 3×3 的 filter 好于 使用一个 7×7 的 filter，3 个 3×3 的 filter对别提高了模型的判别性（discriminative），那么什么是模型的判别性呢？这个好向和模型的类型有关，模型可以分为生成模型和判别模型。是不是想要了解什么是更具判别性，就需要去了解什么是判别模型。

通过查找判别模型的资料，也没有特别理解，怎么样模型就更具有判别性了。这里引用一些可能对理解判别性有用的资料：

1. 出自《[VGGNet 阅读理解 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zziahgf/article/details/79614822)》（不感兴趣可略过）

> 以下是个人的深层思考：网络更深带来了更多变化，更好的特征多样性，就好比是数据增强虽然引来方差是好的，我们想在变化中寻找不变的映射关系。但是，网络更深带来特征更多真的好嘛？我觉得更多的特征和更深的网络，不一定都是有助于、有贡献于正确梯度下降寻找最优或者局部最优的方向，我们真正需要的是可以正确建立映射关系的特征。
> 

下面把资料贴出来。

[参考]

- [CSDN - 生成模型与判别模型](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zouxy09/article/details/8195017)

## **什么是密集评估？**

文中介绍了 multi-crop 和 dense evaluation 两种评估方法，对其中的dense evaluation 不是特别理解，在网上查了资料也还是没有太明白，好像这个方法和 FCN （Fully Convolutional Networks）有着密切的关系，从参考中的资料中也没有明白两个是怎么密切相关的。

这里也有说下 FCN 和 FC，之前对这两个一直很模糊，总是把两者混在一起。FCN 指的是 Fully Convolutional Networks ，是指一种卷积神经网络，但这个网络中全部都是卷积层。不像传统的卷积神经网络（Convolutional Networks），前面基层是卷积层，最后几层就不是卷积层了，而是全连接层，即 FC（Full Connections）更多的指的是连接的方式。

虽然没有搞懂 dense evaluation 是什么意思，但把参考资料列出来，有懂的看到的话希望能给指点下。

[参考]

- [知乎 - VGG神经网络论文中multi-crop evaluation的结论什么意思？](https://www.zhihu.com/question/270988169)
- [CSDN - 深度学习（二十）基于Overfeat的图片分类、定位、检测](https://link.zhihu.com/?target=https%3A//blog.csdn.net/hjimce/article/details/50187881)
- [CNBLOGS - VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 这篇论文](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/yinheyi/p/6233950.html) 参考了《测试网络的过程》

## **什么是 全局池化（Global Average Pooling）？**

论文的附录 B《GENERALISATION OF VERY DEEP FEATURES》提到使用全局平均池化（GAP）方法，那么什么是全局平均池化呢？此概念首先在 NIN（Network In Network） 中提出。

首先，需要知道什么是全局池化（global pooling），它其实指的滑动窗口的大小与整个 feature map 的大小一样，这样一整张feature map 只产生一个值。比如一个 4×4 的 feature map 使用传统的池化方法（2×2 + 2s），那么最终产生的 feature map 大小为 2×2 ，如下图：

![https://pic2.zhimg.com/80/v2-e63b652951f78a058889c57b9a94e379_720w.jpg](https://pic2.zhimg.com/80/v2-e63b652951f78a058889c57b9a94e379_720w.jpg)

而如果使用全局池化的话（4×4 + 1s，大小与 feature map 相同），一个feature map 只产生一个值，即输出为 1×1，如下图：

![https://pic1.zhimg.com/80/v2-7f92a76b60653ae561911d1249da55b4_720w.jpg](https://pic1.zhimg.com/80/v2-7f92a76b60653ae561911d1249da55b4_720w.jpg)

使用全局最大池化示意图

如果前一层有多个feature map 的话，只需要把经过全局池化的结果堆叠起来即可，如下图：

![https://pic2.zhimg.com/80/v2-7fd8b84769f4b65cb69bbca5c5c72161_720w.jpg](https://pic2.zhimg.com/80/v2-7fd8b84769f4b65cb69bbca5c5c72161_720w.jpg)

多 Feature Map 全局池化，依据不同的池化方法（Max or Average）产生生不同的值

上图，如果使用 Average 池化方法，那么就成为 Global Average Pooling，即 GAP。

从而可以总结出，如果输入 feature map 为 W×H×C，那么经过全局池化之后的输出就为 1×1×C。

[参考]

- [StackOverflow - What does global pooling do?](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/42070528/what-does-global-pooling-do)
- [CSDN - what is global average pooling ? 全局平均池化层](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_34650787/article/details/80204873)
- [CSDN - Global Average Pooling全局平均池化的一点理解](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_23304241/article/details/80292859)

什么是图像语义？

![https://pic3.zhimg.com/80/v2-12e46a46e1618ef125515409d58a9792_720w.jpg](https://pic3.zhimg.com/80/v2-12e46a46e1618ef125515409d58a9792_720w.jpg)

GooLeNet 模型可视化的结果，可在参考中找到

"浅层学到的是纹理特征，而深层学到的是语义特征" 。从上图可以看到越是低层学到的月粗糙，即学到的都一些边缘（edges）或则纹理（textures），越是高层越偏向于语义特征。那么什么是语义特征呢？语义指的到底是什么呢？

这里的语义主要用于图像分割领域，这里的语义仍主要指分割出来的物体的类别，从分割结果可以清楚的知道分割出来的是什么物体，比如猫、狗等等。即指物体的类别，如猫、狗就是语义。上图，越是高层的就越能展现语义特征。现在还有一种 instance segmentation 方法，可以可以对同一类别的不同物体进行不同的划分，可以清楚地知道分割出来的左边和右边的两个人不是同一个人。如下图：

![https://pic3.zhimg.com/80/v2-878f106b6822a85738369e0a3d1a0bb6_720w.jpg](https://pic3.zhimg.com/80/v2-878f106b6822a85738369e0a3d1a0bb6_720w.jpg)

semantic segmentation and instance segmentation

- `semantic segmentation` - 只标记语义, 也就是说只分割出`人`这个类来
- `instance segmentation` - 标记实例和语义, 不仅要分割出`人`这个类, 而且要分割出`这个人是谁`, 也就是具体的实例

[参考]

- [Feature Visualization - How neural networks build up their understanding of images](https://link.zhihu.com/?target=https%3A//distill.pub/2017/feature-visualization/)
- [simshang - Fully Convolutional Networks](https://link.zhihu.com/?target=http%3A//simtalk.cn/2016/11/01/Fully-Convolutional-Networks/)
- [知乎 - 图像语义分割和普通的图像分割区别到底在哪呢？](https://www.zhihu.com/question/51567094)

## **图像中的 L1-normalize 与 L2-normalize**

论文的附录部分也提到了图像的 L2-normalize，此 L2 并不是 CNN 中提到的用于解决过拟合的正则化方法，那么图像中的L2-normalize 有指呢？

L1及其 L2的计算公式如下：

L1→xij′=xij∑i=0H−1∑i=0W−1xij

L2→xij′=xij∑i=0H−1∑i=0W−1xij2

其中 xij′ 表示经过 L1或者 L2的值，H 表示图片的高（Height），W 表示宽（Width）， xij 表示图像第 i行 j 列的像素值。如一个 3×3 的图像，使用 L1与 L2的结果如下图：

![https://pic3.zhimg.com/80/v2-80b6903a553a548cb8c226c1f6f89172_720w.jpg](https://pic3.zhimg.com/80/v2-80b6903a553a548cb8c226c1f6f89172_720w.jpg)

L1 和 L2计算

[参考]

- [CSDN - 图像处理中的L1-normalize 和L2-normalize](https://link.zhihu.com/?target=https%3A//blog.csdn.net/a200800170331/article/details/21737741) 里面的公式有问题
- [VGGNet 阅读理解 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zziahgf/article/details/79614822)

## **什么是 IoU？**

IoU （intersection-over-union）是用于评价目标检测（Object Detection）的评价函数，模型简单来讲就是模型产生的目标窗口和原来标记窗口的交叠率。即检测结果(DetectionResult)与 Ground Truth 的交集比上它们的并集，即为检测的准确率 IoU :

IoU=DR∩GTDR∪GT

其中DR=Detection Result ，GT = Ground Truth。

![https://pic3.zhimg.com/80/v2-fdb5c35ddcb5f9784f60f163364a5a62_720w.jpg](https://pic3.zhimg.com/80/v2-fdb5c35ddcb5f9784f60f163364a5a62_720w.jpg)

或者写成如下的公式：

![https://pic1.zhimg.com/80/v2-cd18906af50edce822a9d8023af1a8c0_720w.jpg](https://pic1.zhimg.com/80/v2-cd18906af50edce822a9d8023af1a8c0_720w.jpg)

可以看到 IoU 的值越大，表明模型的准确度越好，IoU = 1 的时候 DR 与 GT 重合。

[参考]

- [CSDN - 目标识别（object detection）中的 IoU（Intersection over Union）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/lanchunhui/article/details/71190055)
- [CSDN - 检测评价函数 intersection-over-union （ IOU ）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/Eddy_zheng/article/details/52126641)

## **遗留问题**

single-class regression, SCR ； per-class regression, PCR 什么意思？

【参考汇总】

- [StackOverflow - What does global pooling do?](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/42070528/what-does-global-pooling-do)
- [CSDN - what is global average pooling ? 全局平均池化层](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_34650787/article/details/80204873)
- [CSDN - Global Average Pooling全局平均池化的一点理解](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_23304241/article/details/80292859)
- [Feature Visualization - How neural networks build up their understanding of images](https://link.zhihu.com/?target=https%3A//distill.pub/2017/feature-visualization/)
- [simshang - Fully Convolutional Networks](https://link.zhihu.com/?target=http%3A//simtalk.cn/2016/11/01/Fully-Convolutional-Networks/)
- [知乎 - 图像语义分割和普通的图像分割区别到底在哪呢？](https://www.zhihu.com/question/51567094)
- [CSDN - 图像处理中的L1-normalize 和L2-normalize](https://link.zhihu.com/?target=https%3A//blog.csdn.net/a200800170331/article/details/21737741)
- **[VGGNet 阅读理解 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zziahgf/article/details/79614822)**
- [CSDN - 目标识别（object detection）中的 IoU（Intersection over Union）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/lanchunhui/article/details/71190055)
- [CSDN - 检测评价函数 intersection-over-union （ IOU ）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/Eddy_zheng/article/details/52126641)
- [知乎 - VGG神经网络论文中multi-crop evaluation的结论什么意思？](https://www.zhihu.com/question/270988169)
- [CSDN - 深度学习（二十）基于Overfeat的图片分类、定位、检测](https://link.zhihu.com/?target=https%3A//blog.csdn.net/hjimce/article/details/50187881)
- [CNBLOGS - VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 这篇论文](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/yinheyi/p/6233950.html)
- [CSDN - 生成模型与判别模型](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zouxy09/article/details/8195017)

参考：

[论文笔记：Very deep convolutional networks for large-scale image recognition（VGG）_snoopy_21的博客-CSDN博客](https://blog.csdn.net/qq_29598161/article/details/106558790)

[VGG论文翻译-Very Deep Convolutional Networks for Large-Scale Image Recognition_alex1801的博客-CSDN博客](https://blog.csdn.net/weixin_34910922/article/details/107050556)

[VGGNet总结及启发-2014_alex1801的博客-CSDN博客](https://blog.csdn.net/weixin_34910922/article/details/107076211)