# 同时识别年龄与性别
基于PyTorch 实现多任务学习，在同时识别年龄与性别。


## 数据集

[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) 数据集，460,723张图片。

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/imdb-wiki-teaser.png)

这里为简洁只使用 IMDB 数据集。

### 年龄分布：

460723张照片为平衡每个年龄最多只保存5000张，清洗后按年龄分布作图可以得到：

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/age.png)

### 样例数据

1 | 2 |
|---|---|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_img.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_img.jpg) |
|性别：男, 年龄：46|性别：女, 年龄：46|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_img.jpg)| ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_img.jpg) |
|性别：男, 年龄：48|性别：男, 年龄：28|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_img.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_img.jpg) |
|$(result_4|性别：女, 年龄：23|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_img.jpg)| ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_img.jpg) |
|性别：女, 年龄：38|性别：女, 年龄：52|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_out.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_out.jpg) |
|性别：男, 年龄：37|性别：女, 年龄：42|
