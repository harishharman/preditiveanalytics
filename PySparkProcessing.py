from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.util import LinearDataGenerator
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy

# Spark setup
sc = SparkContext("local", "PySparkProcessing")
sqlContext = SQLContext(sc)
file_path = "/Users/nabs/STC/SymphonyIRI/POC/Proposed_System/"
# file_path = "/user/root/iripoc/"

# Define csv to data frame function
def csv_to_df(csv_file, schema_str):
  parts = csv_file.map(lambda l: l.split(","))

  fields = [StructField(field_name, StringType(), True) for field_name in schema_str.split()]
  schema = StructType(fields)

  schema_data = sqlContext.createDataFrame(parts, schema)

  return schema_data

# Define extract_month function
def extract_month(date_row):
  import re
  date_str = date_row.asDict()['WeekEnding']
  pattern = re.compile(r'\d+')
  match_obj = pattern.search(date_str)
  if match_obj:
    return int(match_obj.group(0))
  else:
    return None

# Convert DataFrame into LabeledPoint RDD
def dfToLPRDD(data_row):
  dependent_col = 'volume'
  row_dict = data_row.asDict()
  independent_vars = []
  for key in row_dict:
    if key != dependent_col:
      independent_vars.append(row_dict[key])
  return LabeledPoint(row_dict[dependent_col], independent_vars)


# Load Files from HDFS
# Convert to a dataframe, which is a wrapper around a Spark RDD
ppg_df = csv_to_df(sc.textFile(file_path + "ppg_file.csv"), "ppgid UPC")
store_df = csv_to_df(sc.textFile(file_path + "store_file.csv"), "store Region OUTLET")
UPC_df = csv_to_df(sc.textFile(file_path + "UPC_file.csv"), "store WeekEnding Units UPC FeatureOnly DisplayOnly FeatureDisplay MultQty Categoryid volume Price PriU BasePriu BasePrice TrendIndex CatTrendIndex RegPriU RegPrice WeekOfYear HLDY_LA_LG HLDY_LA_HW HLDY_HA_LG HLDY_HA_HW HLDY_TX_LG HLDY_TX_HW HLDY_XM_LG HLDY_XM_HW HLDY_NY_LG HLDY_NY_HW HLDY_SU_LG HLDY_SU_HW HLDY_VA_LG HLDY_VA_HW HLDY_EA_LG HLDY_EA_HW HLDY_ME_LG HLDY_ME_HW HLDY_ID_LG HLDY_ID_HW LogHolidayIndex LogWeekofyearIndex MfrID BrandID Lift62 Lift116 Lift119 Lift164 Lift169 Lift301 Lift343 Lift353 Lift363 Lift369 Lift383 Lift401 Lift413 Lift441 Lift443 Lift482 Lift548 Lift570 Lift572 Lift574 Lift578 Lift598 Lift605 Lift725 Lift726 Lift751 Lift838 Lift857 Lift873 Lift1000 AbsPrice62 AbsPrice116 AbsPrice119 AbsPrice164 AbsPrice169 AbsPrice301 AbsPrice343 AbsPrice353 AbsPrice363 AbsPrice369 AbsPrice383 AbsPrice401 AbsPrice413 AbsPrice441 AbsPrice443 AbsPrice482 AbsPrice548 AbsPrice570 AbsPrice572 AbsPrice574 AbsPrice578 AbsPrice598 AbsPrice605 AbsPrice725 AbsPrice726 AbsPrice751 AbsPrice838 AbsPrice857 AbsPrice873 AbsPrice1000 LogVolume LogUnits Discount LogPriceIndex LogSeason LogPrice LogRegPrice LogBasePrice LogPriu LogRegPriu LogBasePriu LogSpecialPack Intercept")

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("Three files have been loaded into Spark DataFrames: ppg_df, store_df, UPC_df")
# print("ppg_df : ")
# print(ppg_df.show())
# print("\n\n")
# print("store_df : ")
# print(store_df.show())
# print("\n\n")
# print("UPC_df : ")
# print(UPC_df.show())
# print("\n\n")
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")


# Join the three loaded files to create a dataframe
df = UPC_df.join(ppg_df, ['UPC'], 'inner').select('*')
df = df.join(store_df, ['store'], 'inner').select('*')

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("Three files have been joined into a Spark DataFrame: ppg_df, store_df, UPC_df")
# print("joined_df : ")
# print(df.show())
# print("\n\n")
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")


# # Group by certain variables and determine the counts of each
# grouped_data = df.groupBy(df.store, df.ppgid, df.Categoryid)
# gdf = grouped_data.agg({"*" : "count"})

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("This is a dataframe that has been grouped and each group is being counted")
# print(gdf.show())
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")

# # Sort the dataframe
# df = df.sort(df.store.desc(), df.ppgid)

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("This is a dataframe has been sorted")
# print(df.show())
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")



# # Create new column which is the mean monthly price for each upc & store combination
# ## 1. Create month column
# df = df.withColumn('NumericMonth', df.WeekEnding.substr(2,2).cast(IntegerType()))
# ## 2. group by month, upc, store
# grouped_data = df.groupBy(df.NumericMonth, df.store, df.UPC)
# ## 3. mean of price for each group
# df_avg_price = grouped_data.agg({"Price" : "mean"})
# ## 4. add mean column to original dataset
# df = df.join(df_avg_price, ['NumericMonth', 'store', 'UPC'], 'inner').select("*").drop('NumericMonth')
# # create new column log month price for each upc & store id
# df = df.withColumn('LogMonthPrice', log(df['avg(Price)']))

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("A new column has been created from grouped data in the given dataframe")
# print(df.show())
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")


# # Define function to transpose dataframe by taking one column into memory at a time
# def transposedDF(dataframe):
#   for index, col in enumerate(dataframe.columns):
#     new_row = []
#     for element in dataframe.select(col).collect():
#       new_row.append(element.asDict().values()[0])
#     print(new_row)
#     if index == 0:
#       df_transposed = sqlContext.createDataFrame([new_row])
#     else:
#       df_transposed = df_transposed.unionAll(sqlContext.createDataFrame([new_row]))
#   return df_transposed


# # Transpose matrix
# ## Create numeric matrix in memory
# arr_numeric = [[73, 79], [83, 89], [97, 101]]
# ## Convert it to a dataframe
# df_numeric = sqlContext.createDataFrame(arr_numeric)
# ## Call transpose function on Matrix
# transposed_df_numeric = transposedDF(df_numeric)

# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")
# print("Transpose a numeric matrix")
# print("Before transpose:")
# print(df_numeric.show())
# print("After transpose:")
# print(transposed_df_numeric.show())
# print(""" ___  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ___
#   __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__  __)(__
#  (______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)(______)\n""")


# MLib example of linear regression

# lp_rdd = df.select('store','Units','UPC','FeatureOnly','DisplayOnly','FeatureDisplay','MultQty','Categoryid','volume','Price','PriU','BasePriu','BasePrice','TrendIndex','CatTrendIndex','RegPriU','RegPrice','WeekOfYear','HLDY_LA_LG','HLDY_LA_HW','HLDY_HA_LG','HLDY_HA_HW','HLDY_TX_LG','HLDY_TX_HW','HLDY_XM_LG','HLDY_XM_HW','HLDY_NY_LG','HLDY_NY_HW','HLDY_SU_LG','HLDY_SU_HW','HLDY_VA_LG','HLDY_VA_HW','HLDY_EA_LG','HLDY_EA_HW','HLDY_ME_LG','HLDY_ME_HW','HLDY_ID_LG','HLDY_ID_HW','LogHolidayIndex','LogWeekofyearIndex','MfrID','BrandID','Lift62','Lift116','Lift119','Lift164','Lift169','Lift301','Lift343','Lift353','Lift363','Lift369','Lift383','Lift401','Lift413','Lift441','Lift443','Lift482','Lift548','Lift570','Lift572','Lift574','Lift578','Lift598','Lift605','Lift725','Lift726','Lift751','Lift838','Lift857','Lift873','Lift1000','AbsPrice62','AbsPrice116','AbsPrice119','AbsPrice164','AbsPrice169','AbsPrice301','AbsPrice343','AbsPrice353','AbsPrice363','AbsPrice369','AbsPrice383','AbsPrice401','AbsPrice413','AbsPrice441','AbsPrice443','AbsPrice482','AbsPrice548','AbsPrice570','AbsPrice572','AbsPrice574','AbsPrice578','AbsPrice598','AbsPrice605','AbsPrice725','AbsPrice726','AbsPrice751','AbsPrice838','AbsPrice857','AbsPrice873','AbsPrice1000','LogVolume','LogUnits','Discount','LogPriceIndex','LogSeason','LogPrice','LogRegPrice','LogBasePrice','LogPriu','LogRegPriu','LogBasePriu','LogSpecialPack','Intercept','ppgid').map(dfToLPRDD)
# model = LinearRegressionWithSGD.train(lp_rdd)
# # model.save(sc, file_path + "model/lin_reg_model")
# pred = lp_rdd.map(lambda p: (p.label, model.predict(p.features)))
# print(pred.collect())
print("generateLinearInput")
data = LinearDataGenerator.generateLinearInput(0, [1,2,3], [23, 45, 12], [.2, .5, .9], 50, 12314, 1)
print(data)
print("generateLinearRDD")
data = LinearDataGenerator.generateLinearRDD(sc, 50, 10, 1)
print(data)

# coefficients model


