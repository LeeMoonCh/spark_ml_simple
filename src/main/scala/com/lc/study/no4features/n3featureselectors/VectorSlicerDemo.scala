package com.lc.study.no4features.n3featureselectors

import java.util

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

/*
向量切片，对原始特征向量进行切片，通过指定一个下标集合进行获取。
官网的例子也很清晰。不在这里多做讲解。
 */
object VectorSlicerDemo extends App{

  import com.lc.study.spark

  val data = util.Arrays.asList(
    Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
    Row(Vectors.dense(-2.0, 2.3, 0.0))
  )

  val defaultAttr = NumericAttribute.defaultAttr
  val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName) //这一步主要是给特征的每一列设置名称。
  val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

  val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

  val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

  dataset.show()

  dataset.printSchema()

  slicer.setIndices(Array(1)).setNames(Array("f3")) //选择下标2的和名称为f3的列出来。
  // or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))

  val output = slicer.transform(dataset)
  output.show(false)




}
