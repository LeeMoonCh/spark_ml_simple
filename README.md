##介绍
本工程，用自己的理解进行spark官方网站中对于spark.ml的讲解，所依赖
版本为2.4.5，会在每一章节中进行详细注释，以及专业术语解释。

仅以此工程来进行温习。

##Machine Learning Library (MLlib) Guide
MLlib 是spark提供的机器学习包，用来提供可靠易用的生产中机器学习方法。在高层次API中，它提供了如下特性：

    ML Algorithms: 提供一些通用的机器学习算法：归类、聚类、回归、协同过滤等
    Featurization: 特征提取、转换、降维以及特征选择等。
    Pipelines: 提供管道，包含构造、评估、调试算法的管道工具
    Persistence: 可对模型、算法、管道进行持久化
    Utilities: 可用于线性代数、统计学、数据处理等。

需要注意的是，现在的版本将会以DataFrame为数据基础进行API设计，关于基于RDD的API将会处于维护状态。
本工程基于DF也就是spark.ml包进行代码编写以及注释。

