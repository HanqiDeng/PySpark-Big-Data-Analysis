#!/usr/bin/env python
# coding: utf-8


# import modules
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression, NaiveBayes, DecisionTreeClassifier, GBTClassifier 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs



# create sessions
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName("project").getOrCreate()


# load data
data = spark.read.option("header","true").csv("gs://777-may/202302tripdata.csv")
# drop null values
data = data.na.drop()
# drop unwanted columns, change data type, create new columns
data = data.drop("ride_id","start_station_name","end_station_name","start_station_id","end_station_id")\
            .withColumn("start_lat", col("start_lat").cast("double"))\
            .withColumn("start_lng", abs(col("start_lng").cast("double")))\
            .withColumn("end_lat", col("end_lat").cast("double"))\
            .withColumn("end_lng", abs(col("end_lng").cast("double")))\
            .withColumn("start_day", data.started_at[9:2].cast("integer"))\
            .withColumn("start_hour", data.started_at[12:2].cast("integer"))\
            .withColumn("end_day", data.ended_at[9:2].cast("integer"))\
            .withColumn("end_hour", data.ended_at[12:2].cast("integer"))
            
# one hot encoding
bike_indexer = StringIndexer(inputCol="rideable_type", outputCol="bike_index")
label_indexer = StringIndexer(inputCol="member_casual", outputCol="label")
pipeline = Pipeline(stages=[bike_indexer, label_indexer])
data = pipeline.fit(data).transform(data)
data = data.drop("rideable_type","started_at","ended_at","member_casual")


# Support Vector Machines
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data).select("features", "label")
train, test = data.randomSplit([0.8, 0.2])
svm = LinearSVC(maxIter=10, regParam=0.1)
svm_model = svm.fit(train)
predictions = svm_model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
accuracy = evaluator.evaluate(predictions)
print("SVM Accuracy:", accuracy)


# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)
predictions = lr_model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
accuracy = evaluator.evaluate(predictions)
print("Logistic Regression Accuracy:", accuracy)


# Naive Bayes
nb = NaiveBayes(featuresCol="features", labelCol="label")
nb_model = nb.fit(train)
predictions = nb_model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
accuracy = evaluator.evaluate(predictions)
print("Naive Bayes Accuracy:", accuracy)



# Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train)
predictions = dt_model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
accuracy = evaluator.evaluate(predictions)
print("Decision Tree Accuracy:", accuracy)



# Gradient Boosting Trees
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)
param_grid = (ParamGridBuilder()
              .addGrid(gbt.maxDepth, [2, 4, 6])
              .addGrid(gbt.maxBins, [20, 60])
              .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10)
model = cv.fit(train)
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)
print("GBT Accuracy:", accuracy)

