# malicious_filtering_for_ani

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

这个项目旨在创建一个api为ani项目提供服务器端的自动文本过滤功能，主要应用在弹幕，评论上。 
本项目是一个新手玩具项目，欢迎大家提供改进意见和PR。
因为用户99%为中文所以项目语言为中文，代码注释也是中文。

## 项目结构

项目结构基于 cookiecutter-data-science 模板，详情请参见 [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)

```
├── LICENSE            <- 开源许可证
├── Makefile           <- 帮助你快速构建环境和运行项目的 Makefile，初次打开项目请参考这个文件
├── README.md          <- README
├── data
│   ├── internal       <- 来自 ani 服务器的数据源，暂时没有数据，在等待服务器端的开发工作。
│   ├── interim        <- 临时的数据集，用于开发和训练模型。
│   ├── processed      <- 最终规范的数据集，用于部署。（最佳实践：尽量不要在开发时编辑次数据）
│   ├── external_raw   <- 未经处理的第三方原始数据，包含大量的实验性数据。
│   └── results        <- 模型输出的结果，例如预测的标签。
│
├── docs               <- 默认的 mkdocs 项目；详情请参见 mkdocs.org。暂时没有用到，未来需要详细api文档时会用到。
│
├── models             <- 训练的模型和各类checkpoints
│
├── notebooks          <- Jupyternotebook。命名约定是一个数字（用于排序），
│                         创建者的缩写，以及简短的描述，例如：
│                         `1.0-jqp-initial-data-exploration`。请不要push实验性质的notebook。
│
├── references         <- 参看论文和希望共享的外部资料。
│
├── reports            <- 分析文件，报告，以及其他输出。
│   └── figures        <- 用于报告的图形和图表
│
├── requirements.txt   <- 环境文件，例如：
│                         通过 `pip freeze > requirements.txt` 生成
│
├── setup.cfg          <- flake8 的配置文件，falke被用于代码风格检查
│
└── malicious_filtering_for_ani                <- 本项目的源代码。
    │
    ├── __init__.py    <- 使 malicious_filtering_for_ani 成为一个 Python 模块
    │
    ├── data           <- 下载或生成数据的脚本
    │   └── 
    │
    ├── features       <- 将原始数据转化为建模特征的脚本
    │   └── 
    │
    ├── models         <- 训练模型并使用训练好的模型进行预测的脚本
    │   └──           
    │   


```

--------

## 技术细节

