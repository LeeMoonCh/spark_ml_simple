## PipeLine
该章节，我们会讨论机器学习中的管道pipeline.spark中的管道是一套统一的建设在DF之上的高层次API。  
它帮助我们进行机器学习管道的创建以及优化。

该章节目录如下：

1. **pipline中的主要概念**
   1. DataFrame
   2. PipeLine的组件
      1. 转换
      2. Estimators  估计器
      3. 管道组件的配置项
   3. PipeLine
      1. 管道如何工作
      2. 细节解释
   4. 参数
   5. 如何将PipeLine进行加载和保存
2. 代码案例



代码案例的话，我们通过代码注释进行讲解。

### 1. **pipline中的主要概念**

spark的管道概念主要是受到的python中[scikit-learn](http://scikit-learn.org/) 工程的启发进行设计的，它运行永久通过简单的标准将多个算法继承到单一的管道或者工作流中。

- **[`DataFrame`](http://spark.apache.org/docs/latest/ml-pipeline.html#dataframe)**: Spark ML 使用DF当做数据，就是Spark-sql中的DF，例如：一个ML中的DF可能包含这些列：文本, 特征向量, 标签, and 预测.
- **[`Transformer`](http://spark.apache.org/docs/latest/ml-pipeline.html#transformers)**: 转换器，顾名思义，和Spark中的转换算子一样，将数据集从这样子的变成那样子的。例如：ML中的model 可以将一个带有特征向量的DF转换成一个带有预测结果的DF。
- **[`Estimator`](http://spark.apache.org/docs/latest/ml-pipeline.html#estimators)**:  估计器，可以作用到DF上，比如：任何一种机器学习的算法封装都是一种估计器，它可以用来训练DF，并生成一个model。
- **[`Pipeline`](http://spark.apache.org/docs/latest/ml-pipeline.html#pipeline)**: 一个管道可以将多个转换器和估计器进行连接，用来实现复杂也无需求。
- **[`Parameter`](http://spark.apache.org/docs/latest/ml-pipeline.html#parameters)**: 略

#### DataFrame

讲解略

#### PipleLine组件

##### Transformers

转换器，其实你可以理解成spark-core中的转换算子。但是又有些不同，它主要包含两种类型：

特征转换器：比如map，可以将一个文本列转换成一个特征列。

学习模型：也是一种转换器，它作用在一个DF上，并且对特征向量进行标签预测等。

### Estimators

估计器，它是各种算法的抽象概念，比如逻辑回归，K-近邻算法，都可以抽象成一个估计器，它的核心方法是fit(),该方法会作用到一个DF上，并生成一个model，model就是转换其中的学习模型，它是一个转换器。比如：`LogisticRegression `就是一个估计器，它作用在DF上会生成一个`LogisticRegressionModel`模型，用来对数据集形象预测。

### Properties of pipeline components

对于`Transformer.transform()`s and `Estimator.fit()`而言，都是无状态的，在将来，将会添加有状态支持。

每个转换器实例和估计器实例都包含一个全局唯一id，该id会非常有用。

## Pipeline

在机器学习中，通常会使用一系列算法进行学习和预测。比如：一个文本数据的处理流程可能会包含如下步骤：

- 对每个文本数据进行分词
- 将每个单词进行 数值类型的特征向量话
- 使用算法对特征向量进行训练并得到模型，进行最终的结果预测



这些步骤在Spark中我们使用PipeLine进行管理，将多个流程变成转换器和估计器进行组合就形成 了PipeLine

上面的步骤我们用下图进行表示：

![image-20200309213814674](data/image/no3/1.png)



所有的语言表达都是无力的，我们来看一下使用pipeline可以如何简单的进行代码编写：

```scala
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")
val hashingTF = new HashingTF()
  .setNumFeatures(1000)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("features")
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.001)
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)
```

代码很清楚的显示了，在我们知道某一个业务流程时，可以通过各种转换器和生成器来进行pipeline的定义，最后我们是定义了一个stages来控制数据流程的。

最后一步，pipeline.fit()方法是对已知数据进行进行训练，他会产生一个训练模型，该模型用来对未知数据进行预测。也就是下面这部分图：

![image-20200309214642329](/Users/lc/Library/Application Support/typora-user-images/2.png)

我们来看代码：

```scala
// Make predictions on test documents.
model.transform(test)
  .select("id", "text", "probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }
```

调用fit()产生的模型，进行新数据集的预测。

棒棒滴~~~~完结撒花~~~



