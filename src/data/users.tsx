/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* eslint-disable global-require */

import {translate} from '@docusaurus/Translate';
import {sortBy} from '@site/src/utils/jsUtils';

/*
 * ADD YOUR SITE TO THE DOCUSAURUS SHOWCASE
 *
 * Please don't submit a PR yourself: use the Github Discussion instead:
 * https://github.com/facebook/docusaurus/discussions/7826
 *
 * Instructions for maintainers:
 * - Add the site in the json array below
 * - `title` is the project's name (no need for the "Docs" suffix)
 * - A short (≤120 characters) description of the project
 * - Use relevant tags to categorize the site (read the tag descriptions on the
 *   https://docusaurus.io/showcase page and some further clarifications below)
 * - Add a local image preview (decent screenshot of the Docusaurus site)
 * - The image MUST be added to the GitHub repository, and use `require("img")`
 * - The image has to have minimum width 640 and an aspect of no wider than 2:1
 * - If a website is open-source, add a source link. The link should open
 *   to a directory containing the `docusaurus.config.js` file
 * - Resize images: node admin/scripts/resizeImage.js
 * - Run optimizt manually (see resize image script comment)
 * - Open a PR and check for reported CI errors
 *
 * Example PR: https://github.com/facebook/docusaurus/pull/7620
 */

// LIST OF AVAILABLE TAGS
// Available tags to assign to a showcase site
// Please choose all tags that you think might apply.
// We'll remove inappropriate tags, but it's less likely that we add tags.
export type TagType =
  // DO NOT USE THIS TAG: we choose sites to add to favorites
  | 'favorite'
  | 'AI'
  | 'Attention'
  | 'CVPR'
  | 'MultiModality'
  | 'ImageFusion'
  | 'ChatGPT'
  | 'Writing'
  | 'GPT4'
  | 'Tips'
  | 'Transformer'
  | 'ResNet'
  | 'ActiveLearning'
  | 'DiffusionModel'
  | 'RecommendationSystems'
  | 'GAN'

// Add sites to this list
// prettier-ignore
const Users: User[] = [
  {
    title: 'ResNet-RS',
    description: '谷歌领衔调优ResNet，性能全面超越EfficientNet系列',
    preview: require('./showcase/ResNet-RS.png'),
    website: 'https://mp.weixin.qq.com/s/I27DJYnV9RB7P21i9Tt3WQ',
    source: 'https://arxiv.org/abs/2103.07579',
    tags: ['AI','favorite','ResNet'],
  },
  {
    title: 'GRL',
    description: 'CVPR23｜即插即用系列！一种轻量高效的自注意力机制助力图像恢复网络问鼎SOTA',
    preview: require('./showcase/GRL.png'),
    website: 'https://mp.weixin.qq.com/s/eMv3oN515it9V--MgEXCZw',
    source: 'https://arxiv.org/pdf/2303.00748',
    tags: ['AI','Attention','CVPR'],
  },
  {
    title: 'CDDFuse',
    description: 'CVPR 2023 | 结合Transformer和CNN的多任务多模态图像融合方法',
    preview: require('./showcase/CCDFuse.png'),
    website: 'https://mp.weixin.qq.com/s/B9lUUAfLdMgYlBVBw7jRYA',
    source: 'https://arxiv.org/abs/2211.14461',
    tags: ['AI','MultiModality','CVPR', 'ImageFusion', 'favorite', 'Transformer'],
  },
  {
    title: 'ChatGPT润色',
    description: 'ChatGPT 能否替代人工润色，终结高成本润色时代？',
    preview: require('./showcase/ChatGPTModify.png'),
    website: 'https://mp.weixin.qq.com/s/soLndUslSwjOF2q7hDoZug',
    source: 'https://www.citexs.com/Editing',
    tags: ['ChatGPT','Writing'],
  },
  {
    title: '论文Results写作',
    description: '论文中的「Results」部分怎么写？这个小细节千万不能忽略......',
    preview: require('./showcase/result.png'),
    website: 'https://mp.weixin.qq.com/s/HipWd0Y1DS8yb8hiqhhOTA',
    source: 'https://mp.weixin.qq.com/s/HipWd0Y1DS8yb8hiqhhOTA',
    tags: ['Writing'],
  },
  {
    title: '学术科研专用ChatGPT',
    description: '学术科研专用ChatGPT来了！周末刚开源，GitHub斩获3k+星',
    preview: require('./showcase/ChatGPTAcademic.png'),
    website: 'https://mp.weixin.qq.com/s/ZoKLNm2wJpnSuxIwlgrxWg',
    source: 'https://mp.weixin.qq.com/s/ZoKLNm2wJpnSuxIwlgrxWg',
    tags: ['ChatGPT'],
  },
  {
    title: '如何给模型加入先验知识',
    description: '本文通过一个简单的鸟类分类案例来总结了五个给模型加入先验信息的方法...',
    preview: require('./showcase/PrioriKnowledge.png'),
    website: 'https://mp.weixin.qq.com/s/L__F17LmNh8jW6zsJZwcJA',
    source: 'https://zhuanlan.zhihu.com/p/188572028',
    tags: ['AI'],
  },
  {
    title: 'CVPR23 最新 125 篇论文分方向整理',
    description: 'CVPR23 最新 125 篇论文分方向整理｜检测、分割、人脸、视频处理、医学影像、神经网络结构、小样本学习等方向',
    preview: require('./showcase/CVPR23Paper.png'),
    website: 'https://mp.weixin.qq.com/s/zTyytTEvhA-IbPjM6xa9Qw',
    source: 'https://www.cvmart.net/community/detail/7422',
    tags: ['CVPR'],
  },
  {
    title: 'Copilot X重磅发布！',
    description: 'GPT-4杀疯了！Copilot X重磅发布！AI写代码效率10倍提升，码农遭降维打击...',
    preview: require('./showcase/CopilotX.png'),
    website: 'https://mp.weixin.qq.com/s/HIEfHDsPF8EZwKJ61tDv4g',
    source: 'https://github.com/github-copilot',
    tags: ['GPT4'],
  },
  {
    title: 'Softmax 函数和它的误解',
    description: 'Softmax 是一个数学函数，用于对 0 和 1 之间的值进行归一化...',
    preview: require('./showcase/Softmax.png'),
    website: 'https://mp.weixin.qq.com/s/IMzyV3dbWFsgIf14YSd1FQ',
    source: 'https://medium.com/artificialis/softmax-function-and-misconception-4248917e5a1c',
    tags: ['AI'],
  },
  {
    title: '主动学习(Active Learning)介绍',
    description: '计算机视觉中的主动学习(Active Learning)介绍',
    preview: require('./showcase/ActiveLearning.png'),
    website: 'https://mp.weixin.qq.com/s/qAgZa7E3TF466Oxkva1mJA',
    source: 'https://mp.weixin.qq.com/s/qAgZa7E3TF466Oxkva1mJA',
    tags: ['AI','ActiveLearning'],
  },
  {
    title: '如何用ChatGPT搞科研？',
    description: '这位研究僧，GPT-4都发布了，你还在纯人工搞科研吗？',
    preview: require('./showcase/ChatGPTWays.png'),
    website: 'https://mp.weixin.qq.com/s/xzAScUFQ9dFnbaN1dGI7Qg',
    source: 'https://www.zhihu.com/question/583232012/answer/2886779529',
    tags: ['ChatGPT'],
  },
  {
    title: '7个高赞Python库，切勿重复造轮子',
    description: '当第三方库可以帮我们完成需求时，就不要重复造轮子了，整理了GitHub上7个最受好评的Python库，将在你的开发之旅中提供帮助',
    preview: require('./showcase/PythonPackage.png'),
    website: 'https://mp.weixin.qq.com/s/xjynbg4kX0sD-CAdFGzXxw',
    source: 'https://mp.weixin.qq.com/s/xjynbg4kX0sD-CAdFGzXxw',
    tags: ['Tips'],
  },
  {
    title: '交叉熵损失(Cross-entropy)和平方损失(MSE)究竟有何区别？',
    description: '交叉熵损失为什么会优于MSE，本文详细讲解了两者之间的差别以及两者的优缺点',
    preview: require('./showcase/CrossEntropy.png'),
    website: 'https://mp.weixin.qq.com/s/6Lfv5RkA3o62jFbFEURptA',
    source: 'https://mp.weixin.qq.com/s/6Lfv5RkA3o62jFbFEURptA',
    tags: ['AI'],
  },
  {
    title: '微软亚研院最新工作｜DeepMIM：MIM中引入深度监督方法​',
    description: 'DeepMIM旨在网络的浅层加入额外的监督，使得浅层特征的学习更有意义',
    preview: require('./showcase/DeepMIM.png'),
    website: 'https://mp.weixin.qq.com/s/suhOmVh9c4IwKt9kMfCRKA',
    source: 'https://arxiv.org/pdf/2303.08817.pdf',
    tags: ['AI', 'Transformer'],
  },
  {
    title: 'CVPR2023 | 港科大 & 腾讯 AI Lab & 港大联合出品：有趣的动态 3D 场景重建​',
    description: 'CVPR2023 | 港科大 & 腾讯 AI Lab & 港大联合出品：有趣的动态 3D 场景重建',
    preview: require('./showcase/3DReconstruction.png'),
    website: 'https://mp.weixin.qq.com/s/lkF4Nkdfv6qGG3d4a_FRzQ',
    source: 'https://arxiv.org/abs/2303.05312',
    tags: ['AI', 'CVPR'],
  },
  {
    title: 'CVPR2023|Lite DETR​',
    description: '设计一个高效的编码器块,交错更新高级特征(小分辨率特征图)和低级特征(大分辨率特征图),为了更好地融合多尺度特征，开发一种key-aware的可变形注意力来预测更可靠的注意力权重',
    preview: require('./showcase/LiteDETR.png'),
    website: 'https://mp.weixin.qq.com/s/pfUc_Bmi06DD9nQeFF0V7g',
    source: 'https://github.com/IDEA-Research/Lite-DETR',
    tags: ['AI', 'CVPR', 'Transformer', 'Attention'],
  },
  {
    title: 'CVPR2023|BiFormer​',
    description: 'CVPR2023 即插即用系列! | BiFormer: 基于动态稀疏注意力构建高效金字塔网络架构',
    preview: require('./showcase/BiFormer.png'),
    website: 'https://mp.weixin.qq.com/s/7JGIbNL1GZ3rZljCxRGxFg',
    source: 'https://github.com/rayleizhu/BiFormer',
    tags: ['AI', 'CVPR', 'Transformer', 'Attention'],
  },
  {
    title: 'SCI论文中Methods最好写？​',
    description: '别掉以轻心！写「材料和方法」部分时，这些坑千万别踩！',
    preview: require('./showcase/Methods.png'),
    website: 'https://mp.weixin.qq.com/s/sU0zhtAA-6lj2YRj-330PQ',
    source: 'https://mp.weixin.qq.com/s/sU0zhtAA-6lj2YRj-330PQ',
    tags: ['Writing'],
  },
  {
    title: 'AI画手会画手了!Stable Diffusion学会想象,卷趴人类提示工程师​',
    description: '趁我们不注意,AI画手一直在悄悄迭代,最近新推出的Stable Diffusion Reimagine和Midjourney v5功能如此强大,不仅要淘汰人类画师,连提示工程师的饭碗怕是都要丢了。',
    preview: require('./showcase/Midjourney.png'),
    website: 'https://mp.weixin.qq.com/s/EXw_9ssS5f6VA32IXfWMcQ',
    source: 'https://clipdrop.co/stable-diffusion-reimagine',
    tags: ['AI','DiffusionModel'],
  },
  {
    title: 'Stable Diffusion公司新作Gen-1​',
    description: 'Stable Diffusion公司新作Gen-1:基于扩散模型的视频合成新模型,加特效杠杠的!',
    preview: require('./showcase/Gen-1.png'),
    website: 'https://mp.weixin.qq.com/s/X5GrCefKz9hozd8eyT1fJA',
    source: 'https://arxiv.org/pdf/2302.03011',
    tags: ['AI','DiffusionModel'],
  },
  {
    title: 'DSSM,Youtube_DNN,SASRec,PinSAGE…你都掌握了吗?',
    description: '一文总结推荐系统必备经典模型(一),推荐系统是指利用电子商务网站向客户提供商品信息和建议，帮助用户决定应该购买什么产品，模拟销售人员帮助客户完成购买过程的系统。',
    preview: require('./showcase/RecommendationSystems.png'),
    website: 'https://mp.weixin.qq.com/s/vWTP9DrTRkm8pxlXAnjmYg',
    source: 'https://mp.weixin.qq.com/s/vWTP9DrTRkm8pxlXAnjmYg',
    tags: ['AI','RecommendationSystems'],
  },
  {
    title: 'MAV3D (Make-A-Video3D)',
    description: '一行文本,生成3D动态场景:Meta这个「一步到位」模型有点厉害,在最近的一篇论文中,来自Meta的研究者首次提出了可以从文本描述中生成三维动态场景的方法',
    preview: require('./showcase/MAV3D.png'),
    website: 'https://mp.weixin.qq.com/s/BSNIwA8SvQxOjwk-yTPmjA',
    source: 'https://arxiv.org/abs/2301.11280',
    tags: ['AI'],
  },
  {
    title: 'CVPR2023|微软提出RODIN:首个3D扩散模型高质量生成效果,换装改形象一句话搞定!',
    description: '近日,由微软亚研院提出的Roll-out Diffusion Network (RODIN)模型,首次实现了利用生成扩散模型在3D训练数据上自动生成 3D 数字化身(Avatar)的功能',
    preview: require('./showcase/RODIN.png'),
    website: 'https://mp.weixin.qq.com/s/GNyUXkx9YDj3n7Iq133wZA',
    source: 'https://arxiv.org/abs/2212.06135',
    tags: ['AI','DiffusionModel'],
  },
  {
    title: 'GPT-3解数学题准确率升至92.5%!',
    description: 'ChatGPT的文科脑有救了!微软提出MathPrompter,无需微调即可打造「理科」语言模型',
    preview: require('./showcase/MathPrompter.png'),
    website: 'https://mp.weixin.qq.com/s/vhGUZlwsUqSN4zoA5mNEiw',
    source: 'https://arxiv.org/abs/2303.05398',
    tags: ['AI','ChatGPT'],
  },
  {
    title: '孔乙己终结者!GPT-4拿100美元自创业,还要让马斯克下岗',
    description: 'GPT-4引发的新一波革命，把打工人推上了「断头台」。孔乙己的未来在哪里？',
    preview: require('./showcase/GPT4.png'),
    website: 'https://mp.weixin.qq.com/s/npduP_Rr5sngZudWTFoYbw',
    source: 'https://mp.weixin.qq.com/s/npduP_Rr5sngZudWTFoYbw',
    tags: ['AI','GPT4'],
  },
  {
    title: '朱俊彦CVPR新作GigaGAN',
    description: 'GAN的反击:朱俊彦CVPR新作GigaGAN,出图速度秒杀Stable Diffusion',
    preview: require('./showcase/GigaGAN.png'),
    website: 'https://mp.weixin.qq.com/s/bYvrijfdH2wYNl65lX6ywQ',
    source: 'https://arxiv.org/abs/2303.05511',
    tags: ['AI','GAN'],
  },
  {
    title: '元乘象 Chatlmg',
    description: '会看图的「ChatGPT」来了!给张图就能聊天、讲故事、写广告',
    preview: require('./showcase/Chatlmg.png'),
    website: 'https://mp.weixin.qq.com/s/uZiYpKQOxyXaVX_3wNq1DQ',
    source: 'https://mp.weixin.qq.com/s/uZiYpKQOxyXaVX_3wNq1DQ',
    tags: ['AI','ChatGPT'],
  },
  {
    title: '邓嘉团队提出最新力作ParNet',
    description: '12层也能媲美ResNet?邓嘉团队提出最新力作ParNet,ImageNet top1精度直冲80.7%',
    preview: require('./showcase/ParNet.png'),
    website: 'https://mp.weixin.qq.com/s/EuzokZXE6QZ0U1KBU5RAwA',
    source: 'https://arxiv.org/pdf/2110.07641.pdf',
    tags: ['AI'],
  },
  {
    title: '没想到Dropou还藏了一手!不仅可以防止过拟合,也可以减小欠拟合?',
    description: '来自FAIR,Meta AI,UC Berkeley,MBZUAI的研究员们继续对Dropout进行了探索,证明了在训练开始时使用Dropout也可以缓解欠拟合。',
    preview: require('./showcase/Dropout.png'),
    website: 'https://mp.weixin.qq.com/s/ZkCw-4gNDwEbibUUqo9SRQ',
    source: 'https://arxiv.org/pdf/2303.01500.pdf',
    tags: ['AI'],
  },
  {
    title: '深度学习刷SOTA的trick盘点',
    description: '盘点了一些在深度学习中常用的、好用的提升模型效果的trick,建议收藏!',
    preview: require('./showcase/SOTATrick.png'),
    website: 'https://mp.weixin.qq.com/s/WGq1ODy-rFo9IqPb1rQ2jg',
    source: 'https://mp.weixin.qq.com/s/WGq1ODy-rFo9IqPb1rQ2jg',
    tags: ['AI','Tips'],
  },
  {
    title: '谷歌大脑深度学习调参(炼丹)指南出炉,Hinton点赞,一天收获1500星',
    description: '为了破除「迷信」,高举科学旗帜,近日来自谷歌大脑、哈佛大学的研究人员发布了《Deep Learning Tuning Playbook》,旨在帮助大家解决这一AI领域的老大难问题',
    preview: require('./showcase/TuningPlaybook.png'),
    website: 'https://mp.weixin.qq.com/s/X5JVsLXSJs4oCIXgmbkmmg',
    source: 'https://github.com/google-research/tuning_playbook',
    tags: ['AI','Tips'],
  },
  {
    title: '理解并统一14种归因算法,让神经网络具有可解释性',
    description: '理解并统一14种归因算法,让神经网络具有可解释性',
    preview: require('./showcase/AttributionAlgorithm.png'),
    website: 'https://mp.weixin.qq.com/s/G3CM4dwQ5DIODlNAQNPayQ',
    source: 'https://arxiv.org/pdf/2303.01506.pdf',
    tags: ['AI'],
  },
  {
    title: '一文看懂CV中的注意力机制',
    description: '一文看懂CV中的注意力机制',
    preview: require('./showcase/Attention.png'),
    website: 'https://mp.weixin.qq.com/s/Hy_P3-2KTZcfp1cMV7Hxpw',
    source: 'https://mp.weixin.qq.com/s/Hy_P3-2KTZcfp1cMV7Hxpw',
    tags: ['AI','Attention'],
  },


  /*
  Pro Tip: add your site in alphabetical order.
  Appending your site here (at the end) is more likely to produce Git conflicts.
   */
];

export type User = {
  title: string;
  description: string;
  preview: string | null; // null = use our serverless screenshot service
  website: string;
  source: string | null;
  tags: TagType[];
};

export type Tag = {
  label: string;
  description: string;
  color: string;
};

export const Tags: {[type in TagType]: Tag} = {
  favorite: {
    label: translate({message: 'Favorite'}),
    description: translate({
      message:
        'Our favorite news!',
      id: 'showcase.tag.Favorite.description',
    }),
    color: '#e9669e',
  },
  AI: {
    label: translate({message: 'AI'}),
    description: translate({
      message: 'AI is intelligence demonstrated by machines!',
      id: 'showcase.tag.AI.description',
    }),
    color: '#39ca30',
  },
  
  MultiModality: {
    label: translate({message: 'Multi-Modality'}),
    description: translate({
      message: 'Multimodality is the application of multiple literacies within one medium!',
      id: 'showcase.tag.MultiModality.description',
    }),
    color: '#127f82',
  },
  ImageFusion: {
    label: translate({message: 'Image-Fusion'}),
    description: translate({
      message: 'Image fusion refers to the process of combining two or more images into one composite image!',
      id: 'showcase.tag.ImageFusion.description',
    }),
    color: '#fe6829',
  },
  ChatGPT: {
    label: translate({message: 'Chat-GPT'}),
    description: translate({
      message: 'ChatGPT is a language model developed by OpenAI, designed to respond to text-based queries and generate natural language responses!',
      id: 'showcase.tag.ChatGPT.description',
    }),
    color: '#8c2f00',
  },
  Writing: {
    label: translate({message: 'Writing'}),
    description: translate({
      message: 'Writing tips!',
      id: 'showcase.tag.Writing.description',
    }),
    color: '#4267b2',
  },
  Attention: {
    label: translate({message: 'Attention'}),
    description: translate({
      message: 'Attention is a mechanism that allows an algorithm to look at other positions in the input sequence for clues!',
      id: 'showcase.tag.Attention.description',
    }),
    color: '#dfd545',
  },
  CVPR: {
    label: translate({message: 'CVPR'}),
    description: translate({
      message: 'CVPR is an annual conference on computer vision and pattern recognition!',
      id: 'showcase.tag.CVPR.description',
    }),
    color: '#a44fb7',
  },
  GPT4: {
    label: translate({message: 'GPT4'}),
    description: translate({
      message: 'GPT-4 is a large multimodal model created by OpenAI that can generate text that is similar to human speech!',
      id: 'showcase.tag.GPT4.description',
    }),
    color: '#127f82',
  },
  Tips: {
    label: translate({message: 'Tips'}),
    description: translate({
      message: 'Tips!',
      id: 'showcase.tag.Tips.description',
    }),
    color: '#14cfc3',
  },
  Transformer: {
    label: translate({message: 'Transformer'}),
    description: translate({
      message: ' The Transformer model is based solely on attention mechanisms and does not use recurrence or convolutions!',
      id: 'showcase.tag.Transformer.description',
    }),
    color: '#ffcfc3',
  },
  ResNet: {
    label: translate({message: 'ResNet'}),
    description: translate({
      message: 'ResNet network, a technique called skip connections is used.',
      id: 'showcase.tag.ResNet.description',
    }),
    color: '#e9669e',
  },
  ActiveLearning: {
    label: translate({message: 'Active-Learning'}),
    description: translate({
      message: 'Active Learning is an instructional method that engages students in the learning process beyond listening and passive note-taking',
      id: 'showcase.tag.ActiveLearning.description',
    }),
    color: '#dfd545',
  },
  DiffusionModel: {
    label: translate({message: 'Diffusion-Model'}),
    description: translate({
      message: 'Diffusion models work by destroying training data through the successive addition of Gaussian noise, and then learning to recover the data by reversing this noising',
      id: 'showcase.tag.DiffusionModel.description',
    }),
    color: '#14cfc3',
  },
  RecommendationSystems: {
    label: translate({message: 'Recommendation-Systems'}),
    description: translate({
      message: 'Recommendation systems are AI software that recommend products and services to users based on their preferences and choices',
      id: 'showcase.tag.RecommendationSystems.description',
    }),
    color: '#4267b2',
  },
  GAN: {
    label: translate({message: 'GAN'}),
    description: translate({
      message: 'GAN is a type of AI model that consists of two separate neural networks that are pitted against each other in a game-like scenario',
      id: 'showcase.tag.GAN.description',
    }),
    color: '#fe6829',
  },
};

export const TagList = Object.keys(Tags) as TagType[];
function sortUsers() {
  let result = Users;
  // Sort by site name
  result = sortBy(result, (user) => user.title.toLowerCase());
  // Sort by favorite tag, favorites first
  result = sortBy(result, (user) => !user.tags.includes('favorite'));
  return result;
}

export const sortedUsers = sortUsers();
