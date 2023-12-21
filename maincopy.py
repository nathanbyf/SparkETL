from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, split
from pyspark.sql.types import StringType
import pyspark.pandas as ps
import os
import sys
import regex as re
from fractions import Fraction
import pycountry
import duckdb as d


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.appName("PySpark Art Data Application").getOrCreate()

df = spark.read.csv("MetObjects.csv", header=True, inferSchema=True)
df.printSchema()


def clean_dimensions(column):
    return column.upper()


clean_dimensions_udf = udf(lambda x: clean_dimensions(x), StringType())


def clean_alphanumeric(text):
    # Remove anything in () since it tends to be repeated measurement
    bracket_pattern = re.compile(r"\((.*?)\)")
    return re.sub(bracket_pattern, "", str(text))


def clean_dims(text):
    # Catch e.g. 1 1/8
    pattern = re.compile(r"[0-9]+ [0-9]+/[0-9]+")
    # Catch e.g solo 1/2
    pattern_inch = re.compile(r"[0-9]+/[0-9]+")
    matches = re.findall(pattern, text)
    matches_inch = re.findall(pattern_inch, text)
    if matches:
        for x in matches:
            try:
                return int(x.split(" ")[0]) + float(Fraction(x.split(" ")[1]))
            except ZeroDivisionError:
                return int(x.split(" ")[0])
    if matches_inch:
        try:
            return [float(Fraction(x)) for x in matches_inch]
        except ZeroDivisionError:
            return 0


def format_seperator(text):
    return str(text) + ",,"


clean_alphanumeric_udf = udf(lambda text: clean_alphanumeric(text))
clean_dims_udf = udf(lambda text: clean_dims(text))
format_seperator_udf = udf(lambda text: format_seperator(text))


def clean_alphanumeric_transform(df):
    return df.withColumn("Dimensions", clean_alphanumeric_udf("Dimensions"))


def clean_dims_transform(df):
    return df.withColumn("Dimensions", clean_dims_udf("Dimensions"))


def format_seperator_transform(df):
    return df.withColumn("Dimensions", format_seperator_udf("Dimensions"))


df2 = (
    df.transform(clean_alphanumeric_transform)
    .transform(clean_dims_transform)
    .transform(format_seperator_transform)
)

df2.select(col("Dimensions")).show()

df3 = (
    df2.withColumn("height_col", split(df2["Dimensions"], ",").getItem(0))
    .withColumn("width_col", split(df2["Dimensions"], ",").getItem(1))
    .withColumn("length_col", split(df2["Dimensions"], ",").getItem(2))
)


def only_numeric(text):
    number_pattern = re.compile(r"[^\d.]+")
    return re.sub(number_pattern, "", str(text))


only_numeric_udf = udf(lambda text: only_numeric(text))


def only_numeric_transform(df3, dim_col):
    return df3.withColumn(dim_col, only_numeric_udf(dim_col))


df4 = (
    df3.transform(only_numeric_transform, "height_col")
    .transform(only_numeric_transform, "width_col")
    .transform(only_numeric_transform, "length_col")
)

# Use pandas-on-Spark DataFrame
df5 = df4.toPandas()

# Apply further transformations to get dimensions in cm
dim_cols = ["height_col", "width_col", "length_col"]
for col in dim_cols:
    df5[col] = df5[col].apply(lambda x: round(float(x) * 2.54, 1) if x else None)

df5.drop(columns="Dimensions", inplace=True)


def clean_country(col):
    unique_countries = re.split("or|,|\|", col)

    # validate countries
    valid_countries = [country.name for country in pycountry.countries]

    # adhoc, need adding to source
    valid_countries.extend(
        ["England", "Scotland", "Ireland", "Northern Ireland", "Wales"]
    )

    unique_valid_countries = [
        country.strip() for country in unique_countries if country in valid_countries
    ]

    return (
        list(set(unique_valid_countries)) if list(set(unique_valid_countries)) else None
    )


df5["Country"] = df5["Country"].astype(str).apply(lambda x: clean_country(x))

# Forward and back fill constituent ID
df5["Constituent ID"] = df5["Constituent ID"].ffill(limit=1).bfill()

# Write to CSV
# df5.write.format("csv").save("mycsv4/my33")
df5.to_csv("MetObjectsOutput.csv", index=False)


# ANALYTICS
df5["Country"] = df5["Country"].replace("[", "").replace("]", "")

df10 = d.query(
    """
    WITH 
    pieces AS (SELECT Country, count(*) no_of_pieces FROM df5 GROUP BY Country
    ) 
    ,artists AS (SELECT Country, count(distinct 'Artist Display Name') AS distinct_artists FROM df5 GROUP BY Country
    ) 
    ,dims AS (SELECT Country, AVG(height_col) AS average_height, AVG(width_col) AS average_width, AVG(length_col) AS average_length FROM df5 GROUP BY Country
    ) 
    ,distinct_cons AS (
    SELECT Country, count(distinct 'Constituent ID') AS distinct_constituents FROM df5 GROUP BY Country
    ) 

    SELECT p.*, a.distinct_artists, d.average_height, d.average_width, d.average_length, dc.distinct_constituents FROM pieces p
    LEFT JOIN artists a on a.Country = p.country
    LEFT JOIN dims D on D.Country = p.Country
    LEFT JOIN distinct_cons dc on dc.Country = p.Country   
    """
).df()


df10.to_csv("MetAnalysis.csv", index=False)

# Spark having memory issues on my machine- hence completed in duck DB above
df5 = spark.createDataFrame(df5)

df5.createOrReplaceTempView("df5")

# Get the number of artworks per individual country
df6 = spark.sql("select count(*), Country from df5 group by Country")

df6.show()

# Get the number of artists per country
df7 = spark.sql(
    "select Country, count(distinct 'Artist Display Name') from df5 group by Country"
)

df7.show()

# Average height, width, and length per country
df8 = spark.sql(
    "select Country, AVG(height_col), AVG(width_col), AVG(length_col) from df5 group by Country"
)

# Collect a unique list of constituent ids per country
df9 = spark.sql("select Country, distinct 'Constituent ID' from df5 group by Country")
