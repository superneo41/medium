from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from mmlspark import ComputeModelStatistics
from mmlspark import LightGBMRegressor
import numpy as np

'''
the data set "train" used here is a distributed spark dataframe

model_df = (spark
  .read                                                # Call the read method returning a DataFrame
  .option("inferSchema","true")                        # Option to tell Spark to infer the schema
  .option("header","true")                             # Option telling Spark that the file has a header
  .json("/FileStore/tables/model_data.json"))    # Option telling Spark where the file is
'''


lgb = LightGBMRegressor(
  numIterations=50,							#keep it low for tuning
  learningRate=0.1,
  weightCol = "weight_new",     #adding weight
  labelCol = label,
  predictionCol = "pred_"+label
)

paramGrid = (ParamGridBuilder()
             .addGrid(lgb.numLeaves, range(10,200,1))
             .addGrid(lgb.maxDepth, range(1,15,1))
             .addGrid(lgb.baggingFraction, np.arange(0,1,0.01))
             ,addGrid(lgb.featureFraction, np.arange(0,1,0.01))
             .addGrid(lgb.minSumHessianInLeaf, np.arrange(0.0005,0.01,0.0001))
             .addGrid(lgb.lambdaL1, range(0,20000000,1))
             .addGrid(lgb.lambdaL2, range(0,20000000,1))
             .build())

evaluator=RegressionEvaluator(predictionCol="pred_"+label, labelCol=label)
cv = CrossValidator(estimator=lgb, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cvModel_lgb = cv.fit(train)


