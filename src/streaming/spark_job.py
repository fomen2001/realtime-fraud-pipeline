from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel

KAFKA_BOOTSTRAP = "kafka:29092"
TOPIC = "transactions"

PG_URL = "jdbc:postgresql://postgres:5432/fraud"
PG_PROPS = {
    "user": "app",
    "password": "app",
    "driver": "org.postgresql.Driver"
}

MODEL_PATH = "/opt/streaming/model"
ALERT_THRESHOLD = 0.80

schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("ts", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("merchant", StringType(), True),
    StructField("country", StringType(), True),
    StructField("category", StringType(), True),
])

def main():
    spark = (SparkSession.builder
             .appName("FraudStreaming")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    model = PipelineModel.load(MODEL_PATH)

    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
           .option("subscribe", TOPIC)
           .option("startingOffsets", "latest")
           .load())

    parsed = (raw.selectExpr("CAST(value AS STRING) AS json")
              .select(F.from_json("json", schema).alias("d"))
              .select("d.*"))

    # ts en timestamp
    parsed = parsed.withColumn("ts", F.to_timestamp("ts"))

    def write_batch(batch_df, batch_id: int):
        if batch_df.rdd.isEmpty():
            return

        # tx_count par client dans le micro-batch (proxy de "5 minutes")
        counts = (batch_df.groupBy("customer_id")
                  .agg(F.count("*").alias("tx_count_5min")))

        df = (batch_df.join(counts, on="customer_id", how="left")
              .withColumn("hour", F.hour("ts"))
              .withColumn("is_foreign", (F.col("country") != F.lit("FR")).cast("int"))
              .withColumn("amount_log", F.log1p(F.col("amount"))))

        # Scoring par le modèle Spark ML pipeline
        scored = model.transform(df)

        # risk_score = proba classe 1
        from pyspark.ml.functions import vector_to_array
        scored = scored.withColumn("risk_score", vector_to_array("probability")[1])


        # Préparer table transactions_enriched
        enriched = scored.select(
            "transaction_id","customer_id","ts","amount","currency","merchant","country","category",
            "tx_count_5min",
            # zscore proxy (dans batch) : (amount - mean)/std
            ((F.col("amount") - F.avg("amount").over(
                __import__("pyspark").sql.window.Window.partitionBy()
            )) / (F.stddev("amount").over(
                __import__("pyspark").sql.window.Window.partitionBy()
            ))).alias("amount_zscore"),
            "risk_score"
        )

        # Écrire en Postgres (upsert simple via append + PK => si collision, erreur)
        # Pour un vrai upsert, on ferait une table staging + merge côté SQL.
        enriched.write.jdbc(url=PG_URL, table="transactions_enriched", mode="append", properties=PG_PROPS)

        alerts = (scored
                  .filter(F.col("risk_score") >= F.lit(ALERT_THRESHOLD))
                  .withColumn("reason",
                              F.when(F.col("amount") > 800, F.lit("High amount"))
                              .when(F.col("country").isin("RU","NG","CM"), F.lit("Risky country"))
                              .otherwise(F.lit("Model flag")))
                  .withColumn("severity",
                              F.when(F.col("risk_score") >= 0.95, F.lit("HIGH"))
                              .when(F.col("risk_score") >= 0.85, F.lit("MEDIUM"))
                              .otherwise(F.lit("LOW")))
                  .select("transaction_id","ts","risk_score","reason","severity"))

        alerts.write.jdbc(url=PG_URL, table="fraud_alerts", mode="append", properties=PG_PROPS)

    query = (parsed.writeStream
             .foreachBatch(write_batch)
             .outputMode("update")
             .option("checkpointLocation", "/tmp/fraud_checkpoint")
             .start())

    query.awaitTermination()

if __name__ == "__main__":
    main()
