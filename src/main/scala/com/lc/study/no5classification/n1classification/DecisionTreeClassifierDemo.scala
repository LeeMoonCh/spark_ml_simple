package com.lc.study.no5classification.n1classification


import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, RFormula, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.{Vectors,Vector}
import org.apache.spark.sql.SparkSession

/*
决策树分类器。

决策树分类，主要应对与二分类问题。就是这个事情到底会不会发生。其实在生活中，很多问题
都可以分成二分类问题。
比如，判断一个人是否胖不胖，一场会议到底成功不成功。一次营销到底好不好等等。
相对应学习用的数据集，生产中产生的数据可能维度过多。所以在生产中，一般会对数据进行
加工处理。也就是降维或者使用VectorIndexer 进行维度选择。

官网的例子是使用的sample_libsvm_data  数据，一共有600多个维度，全部用来当做决策树的判断因子绝对是
不行的。所以官网对这600多个维度的数据进行了VectorIndexer 的维度选择。从而影响决策树的行为。

https://www.cnblogs.com/beiyan/p/8296852.html
大家可以看这篇文章来理解决策树能干什么。

 */
object DecisionTreeClassifierDemo extends App {

  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  //我们通过官网来看一下，这边sparkml对于决策树而言，该如何进行使用，当然，官网的数据还是太过于复杂，
  //之后我们会经过一个比较好的例子进行学习。
  // 读取数据。
//  val data = spark.read.format("libsvm").load("data/simple_libsvm_data")
//
//  // 对所有数据进行string变小标的训练。
//  val labelIndexer = new StringIndexer()
//    .setInputCol("label")
//    .setOutputCol("indexedLabel")
//    .fit(data)
//  // 自动从数据中获取到用于决策树中的维度。
//  val featureIndexer = new VectorIndexer()
//    .setInputCol("features")
//    .setOutputCol("indexedFeatures")
//    .setMaxCategories(4) // 这里表示当类别大于4的时候 这个维度就会被当做非分类因子。
//    .fit(data)
//
//  // 将数据分割成两份，一份为训练数据，一份为测试数据。
//  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
//
//  // 建立决策树模型。主要设置标签列以及特征列。
//  val dt = new DecisionTreeClassifier()
//    .setLabelCol("indexedLabel")
//    .setFeaturesCol("indexedFeatures")
//
//  // 最后将数据的标签再转换成字符串。
//  val labelConverter = new IndexToString()
//    .setInputCol("prediction")
//    .setOutputCol("predictedLabel")
//    .setLabels(labelIndexer.labels)
//
//  // 这次试用pipeline进行模型创建。
//  val pipeline = new Pipeline()
//    .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//
//  // 用训练数据进行训练。
//  val model = pipeline.fit(trainingData)
//
//  // 对测试数据进行预测。
//  val predictions = model.transform(testData)
//
//  // 看一下预测结果的前5条。
//  predictions.select("predictedLabel", "label", "features").show(5)
//
//  // Select (prediction, true label) and compute test error.  看一下正确率。
//  val evaluator = new MulticlassClassificationEvaluator()
//    .setLabelCol("indexedLabel")
//    .setPredictionCol("prediction")
//    .setMetricName("accuracy")
//  val accuracy = evaluator.evaluate(predictions)
//  println(s"Test Error = ${(1.0 - accuracy)}")
//
//  //最后看一下树模型。
//  val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
//  println(s"Learned classification tree model:\n ${treeModel.toDebugString}")



  //--------------以上为官方例子。只是用来看一下效果。以及写法。
  //--------------以下我们来自己找些数据，进行玩耍。
//  UCI数据集 是机器学习不错的数据集网站，本文选取其中的 Balloons 数据集

  //因为数据全是这种字符串，所以我们的想办法将这些字符串变成数值。
  val df = spark.read.format("csv")
      .option("header",true)
      .load("data/yellow-small.data")


  //使用RFormula进行数据转换。

  val rf = new RFormula()
    .setFormula(" label ~ color + size + action + c ")
    .setFeaturesCol("features")
    .setLabelCol("labelIndex")
    .setHandleInvalid("keep")
    .fit(df)

  val data = rf.transform(df)


  //新建决策数据模型，我们少了一步维度选择这一步，因为数据的所有维度都分类影响因子。
  val dt = new DecisionTreeClassifier()
    .setLabelCol("labelIndex")
    .setFeaturesCol("features")
    .fit(data)

  dt.transform(data).show(false)

//  //然后新建数据集，进行预测。
  val test = rf.transform(Seq(("YELLOW","SMALL","STRETCH","ADULT"))
    .toDF("color","size","action","c"))

  //使用模型进行预测,其实这里使用dt的转换方法也是可以的。
  test.map(f=>{
    dt.predict(f.getAs[Vector]("features"))
  }).show()
































}
