
<p align="center">
![COVID-19-china-solution CI](https://github.com/veaba/covid-19-china-solution/workflows/COVID-19-china-solution%20CI/badge.svg)
<p>

# A COVID-19 china solution（COVID-19——中国方案）

一份关于新冠病毒(COVID-19)中国方案。也许宗教信仰、社会体制、政治制度不尽相同，但个体生命有且仅有一次。

Maybe religious beliefs, social systems, and political systems are not the same, but there are only one individual life.

<p style="text-align: center">
<a href="https://covid-19-china-solution.datav.ai">covid-19-china-solution</a>
</p>

> 本仓库由@covid-19-china-solution 开源社区 整理发布

## 什么叫**COVID-19——中国方案**?

一份由开源社区作者整理中国在应对`COVID-19`（新冠病毒）所采取多种举措汇编而成为的文档，这称呼为COVID-19——中国方案


## 如何参与？

先决条件
- 安装 [Node](https://nodejs.org/)
- 安装 [Python](https://python.org/)
``` bash
# 第一步，clone仓库
git clone https://github.com/veaba/covid-19-china-solution.git
# 第二步，进入到项目目录
cd covid-19-china-solution
# 第三步，命令行安装以来
npm install
# 第四步，命令行中，启动开发环境
npm run dev 
# 第五步，更改你的内容,然后可以通过工具来实现自动化翻译，并提交你的更改
python script/translate.py

```



## i18n

如何为自己的母语增加版本？

- docs/.vuepress/nav copy 一份 en.js 改写 自己母语xx.js
- docs/.vuepress/sidebar copy 一份 en.js 改写 自己母语xx.js
- docs/.vuepress/config.js  在themeConfig 仿写前面的结构
- docs/ copy 一份en 目录，改下自己的母语文件

|简写/Code|Language 语言|
|---------|------------|
| /      | 简体中文|
| en     | 英文|
| es     | 西班牙语|
| arab   | 阿拉伯|
| br     | 葡萄牙|
| farsi  | 波斯语|
| fr     | 法语|
| german | 德语|
| id     | 印度尼西亚语|
| italy  | 意大利语|
| jp     | 日语|
| kr     | 韩语|
| ru     | 俄语|
| vi     | 越南语|


