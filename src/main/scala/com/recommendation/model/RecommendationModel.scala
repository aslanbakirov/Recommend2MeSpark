package com.recommendation.model

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.scheduler.SplitInfo
import org.apache.spark.mllib.recommendation._
/**
 * @author ${user.name}
 */

object RecommendationModel {
  
  val sc = new SparkContext("local[2]", "Recommendation Job")
  
  def arrayToString(a:Array[Double]):String ={
    
    var str="";
  
    for(value : Double <- a) {
    	str += value.+(",");
    }
    return str;
  }
  
  def StringToDoubleArray(s:String):Array[Double] ={
	  
      var arr: List[Double] = List()
      val fields = s.split(",")

      for(i <- 0 to (fields.length - 1)){
	    arr:+=fields(i).toDouble;
  	  }
      return arr.toArray
  }
  
  def builtModel() {

    //val sc = new SparkContext("local[2]", "Recommendation Job")
  
    val ratings = sc.textFile("hdfs://10.100.8.55:8020/foodData/food_user_product_score_int.txt").map { line => val fields = line.split(',')

      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)}.cache()

    val model = ALS.train(ratings, 10, 10, 0.1)
    println("model created....")
   
     model.userFeatures.map(line=> line._1 +":" + arrayToString(line._2)).saveAsTextFile("hdfs://10.100.8.55:8020/foodData/userFeatures");  
     model.productFeatures.map(line=> line._1 +":" + arrayToString(line._2)).saveAsTextFile("hdfs://10.100.8.55:8020/foodData/productFeatures");  
   }
   
   def loadModel():MatrixFactorizationModel={

     
    val userX = sc.textFile("hdfs://10.100.8.55:8020/foodData/userFeatures/part*").map { line => val fields = line.split(":")
     
    new Tuple2(fields(0).toInt, StringToDoubleArray(fields(1)))}.cache();
   
    val productf = sc.textFile("hdfs://10.100.8.55:8020/foodData/productFeatures/part*").cache();
   
    val productX = productf.map{ line => val fields = line.split(":")
     
     new Tuple2(fields(0).toInt,StringToDoubleArray(fields(1)))}.cache();
   
    val model = new MatrixFactorizationModel(10, userX, productX);
    
    
    val productList = sc.textFile("hdfs://10.100.8.55:8020/foodData/productIdRealId.txt")
    
    val productIds = productList.map(_.split(",")(0)).cache()
    
    val userList = sc.textFile("hdfs://10.100.8.55:8020/foodData/sortedUserIds.txt").cache()
    
    //val userIds =  userList.map(_.split(",")(0)).cache
    
    val userRankArray = new Array[Array[Double]](5)
    
    var scoreArray = new Array[Double](5)
    
    val userArray = userList.take(5)
    val productArray = productIds.take(5)
    
    for(i <- 0 to (userArray.length-1)){
    	for(j <- 0 to (productArray.length-1)){
    	  scoreArray(j) = model.predict(userArray(i).toInt,productArray(j).toInt)
    	}
    	userRankArray(i) = scoreArray
    	scoreArray = new Array[Double](5)
    }
    
    
    val recommendationArray = new Array[Array[Double]](5)
    
    for(i<- 0 to (userArray.length-1)){
       recommendationArray(i) = sc.parallelize(userRankArray(i)).top(3)
    }
    
    val recommendationList: List[List[Double]] = List()
    
    for(i<- 0 to (userArray.length-1)){
       recommendationList:+(recommendationArray(i).toList)
    }

    sc.parallelize(recommendationList).saveAsTextFile("hdfs://10.100.8.55:8020/foodData/recommendationList")
    
//    for(i<- 0 to (userArray.length-1)){
//       sc.parallelize(recommendationArray(i)).saveAsTextFile("hdfs://10.100.8.55:8020/foodData/recommendationArray"+i)
//    }
    
    return model;
  }
  
   def predict(userId:Int, productId:Int, model:MatrixFactorizationModel):Double={
   return model.predict(userId,productId);
  }
     
  def main(args: Array[String]) {
    
  }

}
