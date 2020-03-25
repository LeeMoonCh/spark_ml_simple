package com.lc.study.no5classification.n1classification

import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, Imputer, IndexToString, RFormula, StringIndexer, VectorIndexer}
import org.apache.spark.sql.Row

/*
随机森林。
决策树分类算法，可以很好的对类别进行分类，但是，前提条件是我们对该书的叶子，也就是其分类因子进行了减少
并且能保证这些分类因子可以准确的表达分类目标。
但是再生产中，这种数据其实很少。那么就容易造成过拟合问题。过拟合问题就是在训练数据的预测中错误率很低
而在测试数据中错误率很高。
为了解决这个问题，随意森林算法就出现了，它有多个决策树组成。综合其结果，进行分类预测。
其使用过程和决策树的使用过程大差不差。

https://blog.csdn.net/w952470866/article/details/78987265/

 */
object RandomDorestClassifierDemo extends App {

  //同样道理，我们先去看一下官方例子，官方例子也是非常复杂的。使用的数据集很难有特殊含义。
  //之后我们会之后一个数据集进行练习观察。
  import com.lc.study.spark

  import spark.implicits._

  //整体流程其实和之前的决策树一样，毕竟本质上来说随机森林就是非常多个决策树组成的。
  // 加载数据集，还是使用的libsvm数据集
//  val data = spark.read.format("libsvm").load("data/simple_libsvm_data")
//
//  //同样道理，将label标签进行数值转换。
//  val labelIndexer = new StringIndexer()
//    .setInputCol("label")
//    .setOutputCol("indexedLabel")
//    .fit(data)
//
//  //特征选择器，进行具体特征选择。
//  val featureIndexer = new VectorIndexer()
//    .setInputCol("features")
//    .setOutputCol("indexedFeatures")
//    .setMaxCategories(4)
//    .fit(data)
//
//  // 切割数据集为训练数据和测试数据。
//  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
//
//  // 新建随机森林算法。
//  val rf = new RandomForestClassifier()
//    .setLabelCol("indexedLabel")
//    .setFeaturesCol("indexedFeatures")
//    .setNumTrees(10)
//
//  // 再讲标签转成文本标签。
//  val labelConverter = new IndexToString()
//    .setInputCol("prediction")
//    .setOutputCol("predictedLabel")
//    .setLabels(labelIndexer.labels)
//
//  //新建pipeline.
//  val pipeline = new Pipeline()
//    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
//
//  // 训练算法模型。
//  val model = pipeline.fit(trainingData)
//
//  // 对测试数据进行预测。
//  val predictions = model.transform(testData)
//
//  // 看看结果。
//  predictions.select("predictedLabel", "label", "features").show(5)
//
//  //进行数据结果的正确率检测。
//  val evaluator = new MulticlassClassificationEvaluator()
//    .setLabelCol("indexedLabel")
//    .setPredictionCol("prediction")
//    .setMetricName("accuracy")
//  val accuracy = evaluator.evaluate(predictions)
//  println(s"Test Error = ${(1.0 - accuracy)}")
//
//  //最后打印算法模型。
//  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
//  println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

  //因为以上数据没有官方字段说明，所以我们也无法看到太实际的意义。
  //所以我们使用泰坦尼克号的乘客数据，进行算法模型的构造。主要用来判断一个人在什么情况下能成功存活。
  //担任实际意义也不是太大。
  /*
首先看一下该数据集的字段含义：
Survived:0代表死亡，1代表存活
Pclass:乘客所持票类，有三种值(1,2,3)
Name:乘客姓名
Sex:乘客性别
Age:乘客年龄(有缺失)
SibSp:乘客兄弟姐妹/配偶的个数(整数值)
Parch:乘客父母/孩子的个数(整数值)
Ticket:票号(字符串)
Fare:乘客所持票的价格(浮点数，0-500不等)
Cabin:乘客所在船舱(有缺失)
Embark:乘客登船港口:S、C、Q(有缺失)

首先需要判断哪些字段影响存活，也就是有效分类因子。
Pclass 肯定有影响，sex也是。Age应该也会影响，但是需要对age进行区间划分。
sibsp、parch应该也会有影响。
大概应该就这些。

   */
  //因为age有空值，所以它将age转换成了string。我们这里需要对age的空值进行转换。
  val trainData =  spark.read.format("csv")
    .option("header",true)
    .load("data/train.csv")
    .map(f=>{
      var age = 0.0
      val ageA = f.getAs[String]("Age")
      if( StringUtils.isNotBlank(ageA)){
        age = ageA.toDouble
      }else{
        age = 1001
      }
      (f.getString(1).trim.toDouble,
        f.getString(2).trim.toDouble,
        f.getString(4),
        age,
        f.getString(6).trim.toDouble,
        f.getString(7).trim.toDouble)
    }).toDF("label","pclass","sex","age","a","b")


    //将age划分为桶分类
  val bucketizer = new Bucketizer()
    .setInputCol("age")
    .setOutputCol("bucketAge")
    .setSplits(Array(0,18,30,45,60,1000,1002))





  //转换完成之后，我们通过，RFomlar算法将特征聚合成一个特征向量。
  val rf = new RFormula()
    .setFormula("label ~ pclass + sex + bucketAge + a + b")
    .setFeaturesCol("f")
    .setLabelCol("index")
    .setHandleInvalid("keep")
//    .fit(trainData)

  //然后新建 随机森林算法。
  val rfc = new RandomForestClassifier()
    .setFeaturesCol("f")
    .setLabelCol("index")
    .setNumTrees(10)
    .setPredictionCol("predict")

  //训练管道模型
  val p1 = new Pipeline()
      .setStages(Array(bucketizer,rf,rfc))
      .fit(trainData)


  //加载测试数据，进行存活预测。
  val testData = spark.read.format("csv")
    .option("header",true)
    .load("data/test.csv")
    .map(f => {
      var age = 0.0
      val ageA = f.getAs[String]("Age")
      if( StringUtils.isNotBlank(ageA)){
        age = ageA.toDouble
      }else{
        age = 1001
      }
      (
        f.getString(1).trim.toDouble,
        f.getString(3),
        age,
        f.getString(5).trim.toDouble,
        f.getString(6).trim.toDouble)
    }).toDF("pclass","sex","age","a","b")

  p1.transform(testData).show(false)
































}
