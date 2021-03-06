package org.apache.spark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import scala.math._

/**
 * Created by Zehao on 2015/3/9.
 */
object MyPI {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("MyPI")
    //for local
    val conf = new SparkConf().setAppName("MyPI").setMaster("local[*]")
    val spark = new SparkContext(conf)
    val slices = if (args.length > 0) args(0).toInt else 2
    val n = 100000 * slices
    val count = spark.parallelize(1 to n, slices).map { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x * x + y * y < 1) 1 else 0
    }.reduce(_ + _)
    println("Pi is roughly " + 4.0 * count / n)
    spark.stop()
  }
}