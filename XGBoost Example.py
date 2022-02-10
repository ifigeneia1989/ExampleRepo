# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost training

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# Use MLflow to track experiments
mlflow.set_experiment("/Users/ifi.derekli@databricks.com/databricks_automl/21-10-11-17:51-AutoML Classification-54d7029f/AutoML Classification-Experiment-54d7029f")

target_col = "income"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import pandas as pd
  
df = spark.read.table("default.ifi_adult_data")
df_loaded = df.toPandas()
display(df_loaded)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer", SimpleImputer(strategy="mean"))
])

transformers.append(("numerical", numerical_pipeline, ["age", "capital_gain", "capital_loss", "education_num", "fnlwgt", "hours_per_week"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### One-hot encoding
# MAGIC Convert each string column into multiple binary columns.
# MAGIC For each input string column, the number of output columns is equal to the number of unique values in the input column.
# MAGIC This is used for string columns with relatively few unique values.

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers.append(("onehot", one_hot_pipeline, ["education", "marital_status", "native_country", "occupation", "race", "relationship", "sex", "workclass"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training - Validation Split
# MAGIC Split the input data into training and validation data

# COMMAND ----------

from sklearn.model_selection import train_test_split

df_loaded = df.toPandas()
split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=521297532, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/1948206081747073/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display="diagram")

xgbc_classifier = XGBClassifier(
  learning_rate=0.3017496595257401,
  max_depth=2,
  min_child_weight=1,
  subsample=0.3733161348842492,
  random_state=521297532,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", xgbc_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                prefix="val_")
    #mlflow.log_param("input_table", "<i>https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#table/hive_metastore/default/ifi_adult_data</i>")
    #mlflow.log_param("input_table1", "&lt;i&gt;https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#table/hive_metastore/default/ifi_adult_data&lt;/i&gt;")
    #mlflow.log_metric("input_table_metric", rmse)
       
display(pd.DataFrame(xgbc_val_metrics, index=[0]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import xgboost
from mlflow.tracking import MlflowClient

xgbc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(xgbc_temp_dir)
xgbc_client = MlflowClient()
xgbc_model_env_path = xgbc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgbc_temp_dir)
xgbc_model_env_str = open(xgbc_model_env_path)
xgbc_parsed_model_env_str = yaml.load(xgbc_model_env_str, Loader=yaml.FullLoader)

xgbc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

with open(xgbc_model_env_path, "w") as f:
  f.write(yaml.dump(xgbc_parsed_model_env_str))
xgbc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgbc_model_env_path, artifact_path="model")
shutil.rmtree(xgbc_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index)))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register to Model Registry

# COMMAND ----------

model_name = "Ifi_ERM_Example"

model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load from Model Registry and Score

# COMMAND ----------

from pyspark.sql.functions import struct
 
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/{registered_model_version.version}","string")

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))
 
new_data = df.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)
# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
  
display(new_data)
