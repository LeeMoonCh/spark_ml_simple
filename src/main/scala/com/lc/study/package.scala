package com.lc

import org.apache.spark.sql.SparkSession

package object study {

  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")


}
