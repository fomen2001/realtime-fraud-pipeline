import json
import random
import time
from datetime import datetime, timezone
from uuid import uuid4
from kafka import KafkaProducer

TOPIC = "transactions"
BOOTSTRAP = "localhost:9092"

countries = ["FR", "DE", "ES", "IT", "BE", "GB", "US", "NG", "CM"]
categories = ["grocery", "electronics", "travel", "fuel", "restaurant", "health"]
merchants = ["amazon", "carrefour", "uber", "total", "fnac", "booking"]

def make_tx():
    return {
        "transaction_id": str(uuid4()),
        "customer_id": f"C{random.randint(1, 200):04d}",
        "ts": datetime.now(timezone.utc).isoformat(),
        "amount": round(max(1.0, random.gauss(60, 40)), 2),
        "currency": "EUR",
        "merchant": random.choice(merchants),
        "country": random.choice(countries),
        "category": random.choice(categories),
    }

def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
    )
    print(f"Producing to topic={TOPIC} on {BOOTSTRAP} ... Ctrl+C to stop")
    try:
        while True:
            tx = make_tx()

            # inject anomalies sometimes
            if random.random() < 0.03:
                tx["amount"] = round(random.uniform(500, 5000), 2)  # big amount
            if random.random() < 0.02:
                tx["country"] = "RU"  # unexpected country

            producer.send(TOPIC, tx)
            time.sleep(0.1)  # ~10 tx/s
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    main()
