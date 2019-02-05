from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import six
import time
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, SQLTransformer, Bucketizer
#from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col,sum


spark = SparkSession.builder.appName('APP_CREDIT_CARD').getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Find correlation between all fields from start to end in a dataframe and a specific field
def getCorrelation(df, start, end, value):
    print( "Correlation to " + value)
    for i in df.columns[start: end]:
        if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
            print("%20s \t %s" %( i, df.stat.corr(value,i)))
    return

# Reading the file and adding header and removing null rows
def readToDF(filename, schema):
    df = spark.read.csv(filename, schema=schema, header=True)
    df.printSchema()
    df.count()
    # 30000
    df.groupBy('label').count().show()
    # We check for null values
    df_null = df.select(*(F.sum(F.col(c).isNull().cast("Double")).alias(c) for c in df.columns)).toPandas()
    df_null
    df = df.filter(df["ID"].isNotNull())
    return df

# function to see the data and label relation - Helpful to clean data
def getDataLabelCounts(df):
    # function to see the data and label relation - Helpful to clean data
    for col in df.columns[2:12]:
        df.groupby('label', col).count().orderBy(col, 'label').show()

    """
    write to a file

    for col in df.columns[2:12]:
        temp = df.groupby('label', col).count().orderBy(col, 'label')
        #df.groupby('label', col).count().orderBy(col, 'label').show()
        file1 = col+"init.csv"
        temp.toPandas().to_csv(file1, header=True)
    """


# Split file
def splitDF(df):
    train, test = df.randomSplit([0.8, 0.2], seed=5)
    train.groupBy('label').count().show()
    test.groupBy('label').count().show()
    return train, test

def getROC(prediction):
    evaluator = BinaryClassificationEvaluator(labelCol='label')
    roc = evaluator.evaluate(prediction, {evaluator.metricName: 'areaUnderROC'})
    print('Test Area Under ROC: {0}'.format(roc))
    return

def getAccuracy(prediction):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    print("Accuracy = %g" % accuracy)
    prediction.groupBy('label', 'prediction').count().show()
    return


### Functions for preprocessing

def getNewBalAmt(df):
    df = df.withColumn('BAL_AMT1', (df.BILL_AMT1 - df.PAY_AMT1) / df.LIMIT_BAL)
    df = df.withColumn('BAL_AMT2', (df.BILL_AMT2 - df.PAY_AMT2) / df.LIMIT_BAL)
    df = df.withColumn('BAL_AMT3', (df.BILL_AMT3 - df.PAY_AMT3) / df.LIMIT_BAL)
    df = df.withColumn('BAL_AMT4', (df.BILL_AMT4 - df.PAY_AMT4) / df.LIMIT_BAL)
    df = df.withColumn('BAL_AMT5', (df.BILL_AMT5 - df.PAY_AMT5) / df.LIMIT_BAL)
    df = df.withColumn('BAL_AMT6', (df.BILL_AMT6 - df.PAY_AMT6) / df.LIMIT_BAL)
    return df


# Looking at the data, Education level 0,4,5 & 6 have same ratio of payment
def getNewEduCol(df):
    df = df.withColumn('EDUCATION_NEW', F.when(df.EDUCATION < 1, 4).when(df.EDUCATION > 4, 4)
                       .otherwise(df.EDUCATION))
    return df

# Converting age to categorical data
def getNewAgeCol(df):
    age_splits = [0, 20, 22, 26, 36, 46, 55, 60, 200]
    bucketizer = Bucketizer(splits=age_splits, inputCol="AGE", outputCol="AGE_NEW")
    df = bucketizer.transform(df)
    return df

# Normalize the continuous values that we will be using
def getNormCols(df, cols):
    df_desc = df.describe().toPandas()   # Gives the mean, min and max values for each attribute
    df_desc.transpose()
    for col in cols:
        min = float(df_desc[col][3])
        max = float(df_desc[col][4])
        dr = max - min
     #   print(min, max, dr)
        df = df.withColumn(col + '_NEW', F.when((df[col] - min)/dr < 0, 0).otherwise((df[col] - min)/dr))
    return df

def getPipeStagesNew(categorical_cols, numerical_cols): # uses OneHotEncoderEstimator
    pipe_stages = []
    for categorical_col in categorical_cols:
        stringIndexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                         outputCols=[categorical_col + "classVec"])
        pipe_stages += [stringIndexer, encoder]
    assembler_inputs = [c + "classVec" for c in categorical_cols] + numerical_cols
    print(assembler_inputs)
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    pipe_stages += [assembler]
    return pipe_stages

def getPipeStages(categorical_cols, numerical_cols):
    pipe_stages = []
    stringIndexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categorical_cols]
    encoders = [OneHotEncoder(dropLast=False, inputCol=stringIndexer.getOutputCol(),outputCol="{0}_encoded".format(stringIndexer.getOutputCol())) for stringIndexer in stringIndexers]
    pipe_stages = stringIndexers + encoders
    assembler_inputs = [encoder.getOutputCol() for encoder in encoders] + numerical_cols
    print(assembler_inputs)
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    pipe_stages += [assembler]
    return pipe_stages


def getTransformedDF(df, p_stages, selected_cols):
    pipeline = Pipeline(stages=p_stages)
    model_p = pipeline.fit(df)
    df = model_p.transform(df)
 #   df.printSchema()
    df = df.select(selected_cols)
 #   df.printSchema()
    return df



# Reading the i/p file and adding the schema

credit_file = "credit_card_clients.csv"

credit_schema = StructType([
    StructField("ID", IntegerType()),
    StructField("LIMIT_BAL", DoubleType()),
    StructField("SEX", DoubleType()),
    StructField("EDUCATION", DoubleType()),
    StructField("MARRIAGE", DoubleType()),
    StructField("AGE", DoubleType()),
    StructField("PAY_0", DoubleType()),
    StructField("PAY_2", DoubleType()),
    StructField("PAY_3", DoubleType()),
    StructField("PAY_4", DoubleType()),
    StructField("PAY_5", DoubleType()),
    StructField("PAY_6", DoubleType()),
    StructField("BILL_AMT1", DoubleType()),
    StructField("BILL_AMT2", DoubleType()),
    StructField("BILL_AMT3", DoubleType()),
    StructField("BILL_AMT4", DoubleType()),
    StructField("BILL_AMT5", DoubleType()),
    StructField("BILL_AMT6", DoubleType()),
    StructField("PAY_AMT1", DoubleType()),
    StructField("PAY_AMT2", DoubleType()),
    StructField("PAY_AMT3", DoubleType()),
    StructField("PAY_AMT4", DoubleType()),
    StructField("PAY_AMT5", DoubleType()),
    StructField("PAY_AMT6", DoubleType()),
    StructField("label", DoubleType())
])

df_read = readToDF(credit_file, credit_schema)

getCorrelation(df_read, 1, -1, "label")
# getDataLabelCounts(df_read)

#------------------Random Forest - 1 -----------------------

### Preprocessing

df_clean = df_read
feature_columns = df_clean.columns[1:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_clean = assembler.transform(df_clean)

### Split file

train, test = splitDF(df_clean)

### Random Forest
print("1. Random Forest with Original features ")
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=50)
model_rf = rf.fit(train)

prediction_rf = model_rf.transform(test)
getROC(prediction_rf)
getAccuracy(prediction_rf)



#------------------Random Forest - 2 -----------------------
### Preprocessing

df_clean = df_read
df_clean = df_clean.withColumn('BAL_AMT1', (df_clean.BILL_AMT1 - df_clean.PAY_AMT1) / df_clean.LIMIT_BAL)

feature_columns = df_clean.columns[1:12] + df_clean.columns[25:26]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_clean = assembler.transform(df_clean)

### Split file

train, test = splitDF(df_clean)

### Random Forest
print("2. Random Forest with only 1 Calculated Balance Amount")
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=50)
model_rf = rf.fit(train)

prediction_rf = model_rf.transform(test)
getROC(prediction_rf)
getAccuracy(prediction_rf)



#------------------Random Forest - 3 -----------------------
### Preprocessing

df_clean = df_read
df_clean = getNewBalAmt(df_clean)   # Adding all the Balance amounts

feature_columns = df_clean.columns[1:12] + df_clean.columns[25:]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_clean = assembler.transform(df_clean)

### Split file

train, test = splitDF(df_clean)

### Random Forest
print("3. Random Forest with all Calculated Balance Amounts")
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=50)
model_rf = rf.fit(train)

prediction_rf = model_rf.transform(test)
getROC(prediction_rf)
getAccuracy(prediction_rf)


#------------------Random Forest - 4 -----------------------


### Random Forest
print("4. Random Forest with all Calculated Balance Amounts & K-fold Cross Validation")

rf = RandomForestClassifier(labelCol='label', featuresCol='features')
pipeline = Pipeline(stages=[rf])

evaluator = MulticlassClassificationEvaluator()

num_folds = 2
#max_depth = [int(x) for x in np.linspace(20, 30, 3)] #maxDepth can only be 30, linspave gives 5 values b/w min and max
max_depth = [5, 7, 8, 9, 10]
max_bins = [25]
num_trees = [40, 45, 50]
impur = ['gini', 'entropy']

paramGrid_rf = ParamGridBuilder().addGrid(rf.maxBins, max_bins).addGrid(rf.maxDepth, max_depth).addGrid(rf.numTrees, num_trees).addGrid(rf.impurity, impur).build()

start_time = time.time()
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid_rf,
    evaluator=evaluator,
    numFolds=num_folds)

model_rf = crossval.fit(train)

model_rf.avgMetrics
a = model_rf.getEstimatorParamMaps()[np.argmax(model_rf.avgMetrics)]
print("Best parameter for Random Forest : %s" % a)


prediction_rf = model_rf.transform(test)
getROC(prediction_rf)
getAccuracy(prediction_rf)

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))



#------------------ Decision Tree & Naive Bayes - 1 -----------------------


### Preprocessing

df_clean = df_read

df_clean = getNewEduCol(df_clean)
df_clean = getNewAgeCol(df_clean)

normalize_cols = df_clean.columns[1:2] + df_clean.columns[12:24]
df_clean = getNormCols(df_clean, normalize_cols)

# Check the correlation again
#getCorrelation(df_clean, 1, len(df_clean.columns), 'label')

# Selecting columns for assembler
categoric_cols = ['SEX', 'EDUCATION_NEW', 'MARRIAGE', 'AGE_NEW', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numeric_cols = df_clean.columns[-13:]     # all the normalized columns
stages = getPipeStages(categoric_cols, numeric_cols)

# Columns we would like to keep in the data
select_cols = ['label', 'features', 'ID', 'SEX', 'EDUCATION_NEW', 'MARRIAGE','AGE_NEW'] + df_clean.columns[6:12]
df_clean = getTransformedDF(df_clean, stages, select_cols)

### Split file

train, test = splitDF(df_clean)

### Decision Tree

print("1. Decision Tree with Original features transformed")
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=10)
model_dt = dt.fit(train)

prediction_dt = model_dt.transform(test)
getROC(prediction_dt)
getAccuracy(prediction_dt)


### Naive Bayes

print("1. Naive with Original features transformed")
nb = NaiveBayes(labelCol="label", featuresCol="features")
model_nb = nb.fit(train)

prediction_nb = model_nb.transform(test)
getROC(prediction_nb)
getAccuracy(prediction_nb)


### Logistic Regression

print("1. Logistic Regression with Original features transformed")
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
model_lr = lr.fit(train)

prediction_lr = model_lr.transform(test)
getROC(prediction_lr)
getAccuracy(prediction_lr)

#------------------ Decision Tree - 2 -----------------------


### Preprocessing

df_clean = df_read

df_clean = getNewBalAmt(df_clean)   # Adding all the Balance amounts
df_clean = getNewEduCol(df_clean)
df_clean = getNewAgeCol(df_clean)

normalize_cols = df_clean.columns[1:2] + df_clean.columns[-8:-2]
df_clean = getNormCols(df_clean, normalize_cols)

# Check the correlation again
getCorrelation(df_clean, 1, len(df_clean.columns), 'label')

# Selecting columns for assembler
# categorical_cols = df.columns[2:12]
categoric_cols = ['SEX', 'EDUCATION_NEW', 'MARRIAGE', 'AGE_NEW', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numeric_cols = df_clean.columns[-7:]     # all the normalized columns
stages = getPipeStages(categoric_cols, numeric_cols)

# Columns we would like to keep in the data
select_cols = ['label', 'features', 'ID', 'SEX', 'EDUCATION_NEW', 'MARRIAGE','AGE_NEW'] + df_clean.columns[6:12] + df_clean.columns[-7:]
df_clean = getTransformedDF(df_clean, stages, select_cols)

### Split file

train, test = splitDF(df_clean)

### Decision Tree

print("2. Decision Tree with Calculated Balance Amounts and features transformed")
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=10)
model_dt = dt.fit(train)

prediction_dt = model_dt.transform(test)
getROC(prediction_dt)
getAccuracy(prediction_dt)


### Naive Bayes

print("2. Naive Bayes with Calculated Balance Amounts and features transformed")
nb = NaiveBayes(labelCol="label", featuresCol="features")
model_nb = nb.fit(train)

prediction_nb = model_nb.transform(test)
getROC(prediction_nb)
getAccuracy(prediction_nb)


### Logistic Regression

print("2. Logistic Regression with Calculated Balance Amounts and features transformed")
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
model_lr = lr.fit(train)

prediction_lr = model_lr.transform(test)
getROC(prediction_lr)
getAccuracy(prediction_lr)



#------------------ Decision Tree - 3 -----------------------

# Using the same train and test file as before but using entropy instead of gini

### Decision Tree

print("3. Decision Tree with Calculated Balance Amounts and features transformed using Entropy")
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=10, impurity ='entropy')
model_dt = dt.fit(train)

prediction_dt = model_dt.transform(test)
getROC(prediction_dt)
getAccuracy(prediction_dt)


#------------------ Decision Tree - 4 -----------------------


print("4. Decision Tree with Calculated Balance Amounts & K-fold Cross Validation")

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
pipeline = Pipeline(stages=[dt])

evaluator = MulticlassClassificationEvaluator()

num_folds = 2
max_depth = [3, 4, 5, 8]
impur = ['gini', 'entropy']
paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, max_depth).addGrid(dt.impurity, impur).build()


crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid_dt,
    evaluator=evaluator,
    numFolds=num_folds)

start_time = time.time()
model_dt = crossval.fit(train)

model_dt.avgMetrics
a = model_dt.getEstimatorParamMaps()[np.argmax(model_dt.avgMetrics)]
print("Best parameter for Decision Tree : %s" % a)

prediction_dt = model_dt.transform(test)
getROC(prediction_dt)
getAccuracy(prediction_dt)

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))


#------------------ Naive Bayes - 3 -----------------------


print("3. Naive Bayes with Calculated Balance Amounts & K-fold Cross Validation")

nb = NaiveBayes(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[nb])

evaluator = MulticlassClassificationEvaluator()

num_folds = 2
paramGrid_nb = ParamGridBuilder().addGrid(nb.smoothing, [0.8, 1.0, 1.2, 1.5]).build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid_nb,
    evaluator=evaluator,
    numFolds=num_folds)

start_time = time.time()
model_nb = crossval.fit(train)

model_nb.avgMetrics
a = model_nb.getEstimatorParamMaps()[np.argmax(model_nb.avgMetrics)]
print("Best parameter for Naive Bayes : %s" % a)

prediction_nb = model_nb.transform(test)
getROC(prediction_nb)
getAccuracy(prediction_nb)

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))


#------------------ Logistic Regression - 3 -----------------------


print("3. Logistic Regression with Calculated Balance Amounts & K-fold Cross Validation")

lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[lr])

evaluator = MulticlassClassificationEvaluator()

num_folds = 2
reg_param = [0.5, 0.01 , 0.0005, .0001 , .00005 ]
paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam, reg_param).build()


crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid_lr,
    evaluator=evaluator,
    numFolds=num_folds)

start_time = time.time()
model_lr = crossval.fit(train)

model_lr.avgMetrics
a = model_lr.getEstimatorParamMaps()[np.argmax(model_lr.avgMetrics)]
print("Best parameter for Naive Bayes : %s" % a)

prediction_lr = model_lr.transform(test)
getROC(prediction_lr)
getAccuracy(prediction_lr)

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))

#-----------------------------END-------------------------------------------