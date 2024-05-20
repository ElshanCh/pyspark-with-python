################################################################################################################################################
# - PySpark Dataframe 
# - Reading The Dataset
# - Checking the Datatypes of the Column(Schema)
# - Selecting Columns And Indexing
# - Check Describe option similar to Pandas
# - Adding Columns
# - Dropping columns
# - Renaming Columns
################################################################################################################################################

### -----> First of all initialize the Spark Session
from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('Dataframe').getOrCreate()

### -----> Reading dataset
df_pyspark=spark.read.option('header','true').csv('test1.csv',inferSchema=True)
# or
df_pyspark=spark.read.csv('test1.csv',header=True,inferSchema=True)
# in the second version the option parameters are passed directly to .csv function

 
### -----> Check the schema, type
df_pyspark.printSchema()
df_pyspark.dtypes

### -----> Vizualizing the dataframe
df_pyspark.show()

### -----> Fetching specific columns from the dataframe
df_pyspark.select(["column_name_1", "column_name_1"]).show()

### -----> Adding Columns in data frame
df_pyspark=df_pyspark.withColumn('new_column_name',df_pyspark['column_name']+2)

### -----> Drop the columns
df_pyspark=df_pyspark.drop('column_name')

### -----> Rename the columns
df_pyspark.withColumnRenamed('Name','New_Name').show()


################################################################################################################################################
# - Dropping Rows
# - Various Parameter In Dropping functionalities
# - Handling Missing values by Mean, Median And Mode
################################################################################################################################################

### -----> Dropping rows with NULL values
df_pyspark.na.drop()
# and
df_pyspark.na.drop(how="any")
# are equal
# however if ~drop(how="all")~ then the rows with all null values will be droped 
df_pyspark.na.drop(how="all")
# ~thresh=3~ in drop() means that don't delete the rows that have at least 3 non NULL values
df_pyspark.na.drop(how="any",thresh=3)
# ~subset~ in drop() takes a column Name, which then will delete all rows where the selected column has NULL value
df_pyspark.na.drop(how="any",subset=['Age'])
# so the rows where ['Age'] column is null will be droped

### -----> Filling the Missing Value
df_pyspark.na.fill('Missing Values',['Name'])
# as a result the all NULL values in the ['Experience','age'] columns will be filled with 'Missing Values' value
# !!!! Please notice that the data type of the column shoul match with the datatype of the value that i si going to be added
# i.e. 'Missing Values' is a string so the dtype of the ['Name'] should be also string

# If you want to fill the null values with mean, median or mode of the column then do as following
from pyspark.ml.feature import Imputer
imputer = Imputer(
    inputCols=["age", "Experience", "Salary"], 
    outputCols=["{}_imputed".format(c) for c in ["age", "Experience", "Salary"]]
    ).setStrategy("median")

imputer.fit(df_pyspark).transform(df_pyspark)


################################################################################################################################################
# - Filter Operation
# - &,|,==
# - ~
################################################################################################################################################

### -----> Filtering. Salary of the people less than or equal to 20000
df_pyspark.filter("Salary<=20000")
# or
# if you want to filter selected columns
df_pyspark.filter("Salary<=20000").select(['Name','age'])
# or
df_pyspark.filter(df_pyspark['Salary']<=20000)

### -----> &,|,==
# multiple filtering
df_pyspark.filter((df_pyspark['Salary']<=20000) | 
                  (df_pyspark['Salary']>=15000))

### -----> ~
# filtering with NOT operation
df_pyspark.filter(~(df_pyspark['Salary']<=20000))
# this filter will take all but NOT the one given in the condition. 
# So anything that is greater than 20,000 will be given over here.



################################################################################################################################################
# Pyspark GroupBy And Aggregate Functions
################################################################################################################################################
### -----> Groupby
# Grouped to find the maximum salary
df_pyspark.groupBy('Name').sum()
# or
df_pyspark.groupBy('Name').avg()
# or
df_pyspark.groupBy('Departments').count()
# or
df_pyspark.agg({'Salary':'sum'})



################################################################################################################################################
# Pyspark ML
################################################################################################################################################
# "training" is our dataframe
# VectorAssembler basically groups the columns provided in inputCols and creates
# a new feature/column as a vector of combination of the values of the columns that are grouped
from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["age","Experience"],outputCol="Independent Features")
output=featureassembler.transform(training)

# here we select the columns that are needed for the ML purpose
finalized_data=output.select("Independent Features","Salary")

from pyspark.ml.regression import LinearRegression
##train test split
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
# initializing the LinearRegression 
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Salary')
# fitting the data to the model
regressor=regressor.fit(train_data)
# Coefficients
regressor.coefficients
# Intercepts
regressor.intercept
# Prediction
pred_results=regressor.evaluate(test_data)
pred_results.predictions.show()
# meanAbsoluteError and meanSquaredError
pred_results.meanAbsoluteError,pred_results.meanSquaredError