# Databricks notebook source
print("hello")

# COMMAND ----------

# MAGIC %r
# MAGIC print("Hello World!")

# COMMAND ----------

# MAGIC %sql
# MAGIC select 1

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

import logging
import sys

#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.ERROR)

logger = logging.getLogger()

#Testing our Logger

logger.error("Our First ERROR Message")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------


