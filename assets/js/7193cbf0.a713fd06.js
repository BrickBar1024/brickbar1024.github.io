"use strict";(self.webpackChunkbrick_bar=self.webpackChunkbrick_bar||[]).push([[4387],{3842:t=>{t.exports=JSON.parse('{"blogPosts":[{"id":"/00intro","metadata":{"permalink":"/hodgwpodge/00intro","editUrl":"https://github.com/BrickBar1024/brickbar1024.github.io/tree/main/hodgwpodge/00intro.md","source":"@site/hodgwpodge/00intro.md","title":"Intro","description":"Hodgepodge\u8bb0\u5f55\u5404\u79cd\u5947\u602a\u4e71\u4e03\u516b\u7cdf\u7684\u4e1c\u897f\uff0cBricklayers\u81ea\u7531\u53d1\u6325\u7684\u5730\u65b9\uff0c\u6709\u4ec0\u4e48\u5e0c\u671bBricklayers\u5199\u7684\u5185\u5bb9\uff0c\u6b22\u8fcecontact us\ud83d\udc4f","date":"2023-05-22T06:42:48.000Z","formattedDate":"2023\u5e745\u670822\u65e5","tags":[],"readingTime":0.295,"hasTruncateMarker":false,"authors":[],"frontMatter":{},"nextItem":{"title":"\u5e38\u89c1\u8bc4\u4ef7\u6307\u6807","permalink":"/hodgwpodge/Common Evaluation Metrics"}},"content":"Hodgepodge\u8bb0\u5f55\u5404\u79cd\u5947\u602a\u4e71\u4e03\u516b\u7cdf\u7684\u4e1c\u897f\uff0cBricklayers\u81ea\u7531\u53d1\u6325\u7684\u5730\u65b9\uff0c\u6709\u4ec0\u4e48\u5e0c\u671bBricklayers\u5199\u7684\u5185\u5bb9\uff0c\u6b22\u8fcecontact us\ud83d\udc4f\\n\\n`\u6700\u4e0b\u9762\u6709\u5206\u7c7b\u7684tag\uff0c\u53ef\u4ee5\u6839\u636e\u611f\u5174\u8da3\u7684tag\u770b\u5bf9\u5e94\u5185\u5bb9`"},{"id":"Common Evaluation Metrics","metadata":{"permalink":"/hodgwpodge/Common Evaluation Metrics","editUrl":"https://github.com/BrickBar1024/brickbar1024.github.io/tree/main/hodgwpodge/01CommonEvaluationMetrics.md","source":"@site/hodgwpodge/01CommonEvaluationMetrics.md","title":"\u5e38\u89c1\u8bc4\u4ef7\u6307\u6807","description":"ROC\u3001AUC","date":"2023-05-22T06:42:48.000Z","formattedDate":"2023\u5e745\u670822\u65e5","tags":[{"label":"Evaluation","permalink":"/hodgwpodge/tags/evaluation"}],"readingTime":3.275,"hasTruncateMarker":false,"authors":[{"name":"Zhiying Liang","title":"\u6df1\u5ea6\u6f5c\u6c34\u9009\u624b","url":"http://joyceliang.club/","imageURL":"https://github.com/JoyceLiang-sudo.png","key":"Zhiying Liang"}],"frontMatter":{"slug":"Common Evaluation Metrics","title":"\u5e38\u89c1\u8bc4\u4ef7\u6307\u6807","authors":["Zhiying Liang"],"tags":["Evaluation"]},"prevItem":{"title":"Intro","permalink":"/hodgwpodge/00intro"},"nextItem":{"title":"Latex\u5907\u5fd8\u5f55","permalink":"/hodgwpodge/Latex\u5907\u5fd8\u5f55"}},"content":"## ROC\u3001AUC\\n\\n![img](./src/CommonEvaluationMetrics/img.png)\\n\\n- \u6b63\u786e\u7387(`Precision`):\\n\\n  $Precision = \\\\frac{TP}{TP + FP}$\\n\\n- \u771f\u9633\u6027\u7387(True Positive Rate, `TPR`)\uff0c\u7075\u654f\u5ea6(``Sensitivity`)\uff0c\u53ec\u56de\u7387(`Recall`)\uff1a\\n\\n  $Sensitivity = Recall = TPR = \\\\frac{TP}{TP + FN}$\\n\\n- \u771f\u9634\u6027\u7387(True Negative Rate, `TNR`)\uff0c\u7279\u5f02\u5ea6(`Specificity`):\\n\\n  $Specificity = TNR = \\\\frac{TN}{FP + TN}$\\n\\n- \u5047\u9634\u6027\u7387(False Negatice Rate, `FNR`)\uff0c\u6f0f\u8bca\u7387(= 1 - \u7075\u654f\u5ea6)\uff1a\\n\\n  $FNR = \\\\frac{FN}{TP + FN}$\\n\\n- \u5047\u9633\u6027\u7387(False Positice Rate, `FPR`)\uff0c\u8bef\u8bca\u7387(= 1 - \u7279\u5f02\u5ea6)\uff1a\\n\\n  $FPR = \\\\frac{FP}{FP + TN}$\\n\\n- \u9633\u6027\u4f3c\u7136\u6bd4(``Positive Likelihood Ratio (LR+)`` )\uff1a\\n\\n  $LR+ = \\\\frac{TPR}{FPR} = \\\\frac{Sensitivity}{1-Specificity}$\\n\\n- \u9634\u6027\u4f3c\u7136\u6bd4(``Negative Likelihood Ratio (LR-)`` )\uff1a\\n\\n  $LR- = \\\\frac{FNR}{TNR} = \\\\frac{1-Sensitivity}{Specificity}$\\n\\n- Youden\u6307\u6570(`Youden index`):\\n\\n  Youden index = Sensitivity + Specificity - 1 = TPR - FPR\\n\\n  \\n\\n\u5982\u4e0b\u9762\u8fd9\u5e45\u56fe\uff0c(a)\u56fe\u4e2d\u5b9e\u7ebf\u4e3aROC\u66f2\u7ebf\uff0c\u7ebf\u4e0a\u6bcf\u4e2a\u70b9\u5bf9\u5e94\u4e00\u4e2a\u9608\u503c\\n\\n![img](././src/CommonEvaluationMetrics/img6.png)\\n\\n(a) \u7406\u60f3\u60c5\u51b5\u4e0b\uff0cTPR\u5e94\u8be5\u63a5\u8fd11\uff0cFPR\u5e94\u8be5\u63a5\u8fd10\u3002ROC\u66f2\u7ebf\u4e0a\u7684\u6bcf\u4e00\u4e2a\u70b9\u5bf9\u5e94\u4e8e\u4e00\u4e2athreshold\uff0c\u5bf9\u4e8e\u4e00\u4e2a\u5206\u7c7b\u5668\uff0c\u6bcf\u4e2athreshold\u4e0b\u4f1a\u6709\u4e00\u4e2aTPR\u548cFPR\\n\\n\u6bd4\u5982threshold\u6700\u5927\u65f6\uff0cTP=FP=0\uff0c\u5bf9\u5e94\u4e8e\u539f\u70b9\uff1bthreshold\u6700\u5c0f\u65f6\uff0cTN=FN=0\uff0c\u5bf9\u5e94\u4e8e\u53f3\u4e0a\u89d2\u7684\u70b9(1, 1)\\n\\n- \u6a2a\u8f74FPR\uff1a1-TNR\uff0c1-Specificity\uff0cFPR\u8d8a\u5927\uff0c\u9884\u6d4b\u6b63\u7c7b\u4e2d\u5b9e\u9645\u8d1f\u7c7b\u8d8a\u591a\\n- \u7eb5\u8f74TPR\uff1aSensitivity(\u6b63\u7c7b\u8986\u76d6\u7387)\uff0cTPR\u8d8a\u5927\uff0c\u9884\u6d4b\u6b63\u7c7b\u4e2d\u5b9e\u9645\u6b63\u7c7b\u8d8a\u591a\\n- \u7406\u60f3\u76ee\u6807\uff1aTPR=1\uff0cFPR=0\uff0c\u5373\u56fe\u4e2d(0, 1)\u70b9\uff0c\u6545ROC\u66f2\u7ebf\u8d8a\u9760\u62e2(0, 1)\u70b9\uff0c\u8d8a\u504f\u79bb45\u5ea6\u5bf9\u89d2\u7ebf\u8d8a\u597d\uff0cSensitivity\u3001Specificity\u8d8a\u5927\u6548\u679c\u8d8a\u597d\\n\\n(b) P\u548cN\u5f97\u5206\u4e0d\u4f5c\u4e3a\u7279\u5f81\u95f4\u8ddd\u79bbd\u7684\u4e00\u4e2a\u51fd\u6570\uff0c\u968f\u7740\u9608\u503ctheta\u589e\u52a0\uff0cTP\u548cFP\u90fd\u589e\u52a0\\n\\n![img](./src/CommonEvaluationMetrics/img3.png)\\n\\n\\n\\n[\u673a\u5668\u5b66\u4e60\u4e4b\u5206\u7c7b\u6027\u80fd\u5ea6\u91cf\u6307\u6807 : ROC\u66f2\u7ebf\u3001AUC\u503c\u3001\u6b63\u786e\u7387\u3001\u53ec\u56de\u7387](https://zhwhong.cn/2017/04/14/ROC-AUC-Precision-Recall-analysis/)\\n\\n[\u6a21\u578b\u8bc4\u4f30\u6307\u6807AUC\uff08area under the curve\uff09_Webbley\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_auc\u6307\u6807](https://blog.csdn.net/liweibin1994/article/details/79462554)\\n\\n## Delong\u2019s test\\n\\n\u6bd4\u8f83AUC\u663e\u8457\u6027\u5dee\u5f02\\n\\n[Delong test_liuqiang3\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_delong\u68c0\u9a8c](https://blog.csdn.net/liuqiang3/article/details/102866673)\\n\\n## McNemar\u2019s test\\n\\n[\u5982\u4f55\u8ba1\u7b97McNemar\u68c0\u9a8c\uff0c\u6bd4\u8f83\u4e24\u79cd\u673a\u5668\u5b66\u4e60\u5206\u7c7b\u5668 - \u817e\u8baf\u4e91\u5f00\u53d1\u8005\u793e\u533a-\u817e\u8baf\u4e91](https://cloud.tencent.com/developer/article/1178350)\\n\\n## C-index\\n\\n[\u4e34\u5e8a\u7814\u7a76\u4e2d\u5e38\u7528\u7684\u8bc4\u4ef7\u6307\u6807AUC\u548cC-index](https://zhuanlan.zhihu.com/p/383272878)\\n\\n[Topic 13. \u4e34\u5e8a\u9884\u6d4b\u6a21\u578b\u4e4b\u4e00\u81f4\u6027\u6307\u6570 (C-index)](https://zhuanlan.zhihu.com/p/485401349)\\n\\n## P-value\u3001confidence interval\\n\\n[\u663e\u8457\u6027\u68c0\u9a8c\uff1aP\u503c\u548c\u7f6e\u4fe1\u5ea6_Chipei Kung\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_\u663e\u8457\u6027\u68c0\u9a8cp\u503c](https://blog.csdn.net/yu1581274988/article/details/117295802)\\n\\n[\u7075\u654f\u5ea6\u548c\u7279\u5f02\u5ea6\u7684\u7f6e\u4fe1\u533a\u95f4\u600e\u4e48\u7b97\uff1f_mjiansun\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_\u654f\u611f\u6027\u7f6e\u4fe1\u533a\u95f4](https://blog.csdn.net/u013066730/article/details/120760183)\\n\\n![img](./src/CommonEvaluationMetrics/img4.png)\\n\\n![img](./src/CommonEvaluationMetrics/img5.png)\\n\\n## Non-inferiority\\n\\n[\u4e34\u5e8a\u8bd5\u9a8c\u4e2d\u5982\u4f55\u9009\u62e9\u975e\u52a3\u6548\u754c\u503c](https://zhuanlan.zhihu.com/p/400409860)\\n\\n## T-test\\n\\n[t-test](https://zhuanlan.zhihu.com/p/38243421)\\n\\n## Kappa\u6307\u6570\\n\\n[kappa\u7cfb\u6570_\u767e\u5ea6\u767e\u79d1](https://baike.baidu.com/item/kappa%E7%B3%BB%E6%95%B0/9385025)\\n\\n[Kappa\u7cfb\u6570\u7b80\u5355\u4ecb\u7ecd_gltangwq\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_kappa\u503c](https://blog.csdn.net/gltangwq/article/details/106357443)\\n\\n## \u591a\u5206\u7c7bAUC\\n\\n[\u591a\u5206\u7c7bROC\u66f2\u7ebf\u53caAUC\u8ba1\u7b97_\u811a\u8e0f\u5b9e\u5730\u4ef0\u671b\u661f\u7a7a\u7684\u535a\u5ba2-CSDN\u535a\u5ba2_\u591a\u5206\u7c7broc\u66f2\u7ebf](https://blog.csdn.net/u010505915/article/details/106450150)"},{"id":"Latex\u5907\u5fd8\u5f55","metadata":{"permalink":"/hodgwpodge/Latex\u5907\u5fd8\u5f55","editUrl":"https://github.com/BrickBar1024/brickbar1024.github.io/tree/main/hodgwpodge/02Latex.md","source":"@site/hodgwpodge/02Latex.md","title":"Latex\u5907\u5fd8\u5f55","description":"\u3010\u8f6c\u3011LaTeX \u7b26\u53f7\u547d\u4ee4\u5927\u5168","date":"2023-05-22T06:42:48.000Z","formattedDate":"2023\u5e745\u670822\u65e5","tags":[{"label":"Memo","permalink":"/hodgwpodge/tags/memo"}],"readingTime":0.76,"hasTruncateMarker":false,"authors":[{"name":"Zhiying Liang","title":"\u6df1\u5ea6\u6f5c\u6c34\u9009\u624b","url":"http://joyceliang.club/","imageURL":"https://github.com/JoyceLiang-sudo.png","key":"Zhiying Liang"}],"frontMatter":{"slug":"Latex\u5907\u5fd8\u5f55","title":"Latex\u5907\u5fd8\u5f55","authors":["Zhiying Liang"],"tags":["Memo"]},"prevItem":{"title":"\u5e38\u89c1\u8bc4\u4ef7\u6307\u6807","permalink":"/hodgwpodge/Common Evaluation Metrics"}},"content":"**[\u3010\u8f6c\u3011LaTeX \u7b26\u53f7\u547d\u4ee4\u5927\u5168](https://www.cnblogs.com/Coolxxx/p/5982439.html)**\\n\\n1. \u628a\u4e0b\u6807\u653e\u5728\u67d0\u4e2a\u6587\u5b57\u6216\u8005\u7b26\u53f7\u6b63\u4e0b\u65b9 \\\\limits\\n   \\n    \u7b26\u53f7\u662f\u6570\u5b66\u7b26\u53f7\uff1a\\\\sum\\\\limits _{i=0}^n {x_i}     $\\\\Rightarrow$    \\n    \\n    $\\\\sum\\\\limits_{i=0}^n {x_i}$\\n    \\n    \u4e0d\u662f\u6570\u5b66\u7b26\u53f7\uff1a \\\\mathop{argmin}\\\\limits _{w,b} L(w,b)    $\\\\Rightarrow$ \\n    \\n     $\\\\mathop{argmin}\\\\limits_{w,b} L(w,b)$\\n    \\n2. \u504f\u5bfc\u7b26\u53f7\uff1a\\\\partial\\n   \\n    \\\\frac{**\\\\partial** }{**\\\\partial** w}     $\\\\Rightarrow$    $\\\\frac{\\\\partial }{\\\\partial w}$  \\n    \\n3. \u7bad\u5934\\n   \\n    **\\\\leftarrow**  $\\\\leftarrow$\\n    \\n    **\\\\rightarrow**  $\\\\rightarrow$\\n    \\n    **\\\\Leftrightarrow** $\\\\Leftrightarrow$\\n    \\n    **\\\\leftrightarrow** $\\\\leftrightarrow$ \\n    \\n    **\\\\Rightarrow ** $\\\\Rightarrow$\\n    \\n4. \u62ec\u53f7\\n   \\n    ( \\\\frac{1}{2} )    $\\\\Rightarrow$     $( \\\\frac{1}{2} )$\\n    \\n    **\\\\left**( \\\\frac{1}{2} **\\\\right**)    $\\\\Rightarrow$     $\\\\left( \\\\frac{1}{2} \\\\right)$ \\n    \\n5. \u77e9\u9635\\n   \\n    **\\\\begin{matrix}** w \\\\\\\\\\\\ b \\\\end{matrix}**    $\\\\Rightarrow$     $\\\\begin{matrix} w \\\\\\\\ b \\\\end{matrix}$ \\n    \\n6. \u53d6\u6574\u51fd\u6570/\u53d6\u9876\u51fd\u6570\\n   \\n    \\\\left **\\\\lfloor** \\\\frac{a}{b} \\\\right\xa0**\\\\rfloor**   ** $\\\\Rightarrow$**    $\\\\left\\\\lfloor \\\\frac{a}{b} \\\\right \\\\rfloor$\\n    \\n    \\\\left **\\\\lceil**\\\\frac{c}{d} \\\\right\xa0**\\\\rceil**    **$\\\\Rightarrow$**    $\\\\left \\\\lceil\\\\frac{c}{d} \\\\right \\\\rceil$\\n    \\n7. \u5927\u4e8e\u7b49\u4e8e\u3001\u5c0f\u4e8e\u7b49\u4e8e\\n\\n   \\\\geq \u3001\\\\leq   $\\\\Rightarrow$ $\\\\geq$  $\\\\leq$\\n\\n8."}]}')}}]);