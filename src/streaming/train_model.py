from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression

MODEL_PATH = "/opt/streaming/model"

def main():
    spark = (SparkSession.builder
             .appName("TrainFraudModel")
             .getOrCreate())

    # ---- Génération data synthétique
    n = 300000  # tu peux monter à 1M si ta machine tient
    countries = ["FR","DE","ES","IT","BE","GB","US","NG","CM","RU"]
    categories = ["grocery","electronics","travel","fuel","restaurant","health"]
    merchants = ["amazon","carrefour","uber","total","fnac","booking"]

    df = (spark.range(n)
          .withColumn("transaction_id", F.expr("uuid()"))
          .withColumn("customer_id", F.concat(F.lit("C"), F.lpad((F.rand()*200).cast("int"), 4, "0")))
          .withColumn("ts", F.current_timestamp())
          .withColumn("amount", F.round(F.greatest(F.lit(1.0), (F.randn()*40 + 60)), 2))
          .withColumn("country", F.element_at(F.array(*[F.lit(x) for x in countries]), (F.rand()*len(countries)+1).cast("int")))
          .withColumn("category", F.element_at(F.array(*[F.lit(x) for x in categories]), (F.rand()*len(categories)+1).cast("int")))
          .withColumn("merchant", F.element_at(F.array(*[F.lit(x) for x in merchants]), (F.rand()*len(merchants)+1).cast("int")))
         )

    # Injecter anomalies
    df = df.withColumn(
        "amount",
        F.when(F.rand() < 0.03, F.round(F.rand()*4500 + 500, 2)).otherwise(F.col("amount"))
    ).withColumn(
        "country",
        F.when(F.rand() < 0.02, F.lit("RU")).otherwise(F.col("country"))
    )

    # Features simples
    df = (df
          .withColumn("hour", F.hour("ts"))
          .withColumn("is_foreign", (F.col("country") != F.lit("FR")).cast("int"))
          .withColumn("amount_log", F.log1p(F.col("amount")))
          .withColumn("tx_count_5min", (F.rand()*10).cast("int"))  # proxy (en réel on le calc en streaming)
         )

    # Label fraude (règle "ground truth" pour entraîner)
    # Ici on crée une fraude si (gros montant) OU (pays inhabituel + fréquence élevée)
    df = df.withColumn(
        "label",
        (
            (F.col("amount") > 800) |
            ((F.col("country").isin("RU","NG","CM")) & (F.col("tx_count_5min") >= 6)) |
            ((F.col("category") == "travel") & (F.col("amount") > 400))
        ).cast("int")
    )

    # ---- Pipeline ML
    cat_cols = ["country", "category", "merchant"]
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    ohe = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]

    feature_cols = ["amount", "amount_log", "tx_count_5min", "is_foreign", "hour"] + [f"{c}_ohe" for c in cat_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="label", probabilityCol="probability", maxIter=50, regParam=0.01)

    pipeline = Pipeline(stages=idx + ohe + [assembler, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)

    # Petite évaluation rapide
    pred = model.transform(test)
    from pyspark.ml.functions import vector_to_array

    pred2 = pred.withColumn("p", vector_to_array("probability")[1])
    acc = pred2.select(
    F.avg(F.when(((F.col("p") >= 0.5) & (F.col("label") == 1)) | ((F.col("p") < 0.5) & (F.col("label") == 0)), 1).otherwise(0)).alias("acc")
  )
    print("Approx ACC:", acc.collect()[0]["acc"])

           

    # Sauvegarde modèle
    model.write().overwrite().save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()
