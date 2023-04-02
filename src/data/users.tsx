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
  | 'SelfAttention'
  | 'CVPR'
  | 'MultiModality'
  | 'ImageFusion'
  | 'ChatGPT'
  | 'Writing'
  | 'GPT4'
  | 'Treasury'
  | 'Transformer'

// Add sites to this list
// prettier-ignore
const Users: User[] = [
  {
    title: 'ResNet-RS',
    description: '谷歌领衔调优ResNet，性能全面超越EfficientNet系列',
    preview: require('./showcase/ResNet-RS.png'),
    website: 'https://mp.weixin.qq.com/s/I27DJYnV9RB7P21i9Tt3WQ',
    source: 'https://arxiv.org/abs/2103.07579',
    tags: ['AI','favorite'],
  },
  {
    title: 'GRL',
    description: 'CVPR23｜即插即用系列！一种轻量高效的自注意力机制助力图像恢复网络问鼎SOTA',
    preview: require('./showcase/GRL.png'),
    website: 'https://mp.weixin.qq.com/s/eMv3oN515it9V--MgEXCZw',
    source: 'https://arxiv.org/pdf/2303.00748',
    tags: ['AI','SelfAttention','CVPR'],
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
    preview: require('./showcase/Result.png'),
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
    tags: ['AI'],
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
    tags: ['Treasury'],
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
    title: '微软亚洲研究院最新工作｜DeepMIM：MIM中引入深度监督方法​',
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
  SelfAttention: {
    label: translate({message: 'Self-Attention'}),
    description: translate({
      message: 'Self-attention is a mechanism that allows an algorithm to look at other positions in the input sequence for clues!',
      id: 'showcase.tag.SelfAttention.description',
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
  Treasury: {
    label: translate({message: 'Treasury'}),
    description: translate({
      message: 'Treasury!',
      id: 'showcase.tag.Treasury.description',
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
