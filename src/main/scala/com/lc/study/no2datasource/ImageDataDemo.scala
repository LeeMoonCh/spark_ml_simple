package com.lc.study.no2datasource

import org.apache.spark.sql.SparkSession

/**
 * 对应图片数据，spark这边的做法是将图片放到目录中，然后使用spark进行目录加载。
 * spark直接可以将format设置为image 就可以加载图片数据了。
 * 加载的列名：image ，该列是一个复合数据类型，它包好了如下字段：
 * origin: StringType (图片路径)
 * height: IntegerType (图片高度)
 * width: IntegerType (图片宽度)
 * nChannels: IntegerType (图像通道数)
 * mode: IntegerType (OpenCV-compatible)
 * data: BinaryType (Image bytes in OpenCV-compatible order:大多数情况是BGR (蓝、绿、红))
 */
object ImageDataDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._



  //是通过 ImageDataSource 类进行数据加载的。
  //在该类的源码注释中我们可以发现这句话:
  //  Image data source supports the following options:
  // *  - "dropInvalid": Whether to drop the files that are not valid images from the result.
  //意思是，提供一个option选项，该选项将无效图片进行删除，而不进行加载。


  val imageData= spark.read.format("image").option("dropInvalid",true)
    .load("data/image")

  imageData.select("image.origin","image.height","image.width","image.nChannels"
    ,"image.mode","image.data").show()  //看下数据集。




}
