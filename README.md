# realtime-fraud-pipeline

---

# 🚀 Pipeline de Détection de Fraude en Temps Réel

## 📌 Vue d’ensemble

Ce projet met en œuvre un **pipeline de détection de fraude en temps réel** en utilisant des technologies modernes de Data Engineering et de Machine Learning.

Le système simule des transactions financières, les traite en temps réel via **Apache Kafka et Spark Structured Streaming**, applique un **modèle de Machine Learning pour le scoring de fraude**, stocke les données enrichies dans **PostgreSQL**, puis visualise les indicateurs via un **dashboard Streamlit**.

Ce projet démontre :

* L’ingestion de données en temps réel
* Le traitement distribué de flux (stream processing)
* L’industrialisation d’un modèle ML
* Une architecture orientée production
* L’aide à la décision dans un contexte métier

---

## 🏗 Architecture

```
Producteur (Python)
        ↓
Kafka (streaming d'événements)
        ↓
Spark Structured Streaming
        ↓
Modèle ML (Scoring fraude)
        ↓
PostgreSQL
        ↓
Dashboard Streamlit
```

---

## ⚙️ Stack Technique

* **Python**
* **Apache Kafka**
* **Apache Spark 3.5 (Structured Streaming + MLlib)**
* **PostgreSQL**
* **Docker / Docker Compose**
* **Streamlit**

---

## 🎯 Fonctionnement du Projet

### 1️⃣ Simulation des Transactions

Un producteur Python génère des transactions synthétiques comprenant :

* montant
* commerçant
* pays
* timestamp
* identifiant client

Ces transactions sont envoyées en temps réel dans Kafka.

---

### 2️⃣ Traitement en Temps Réel

Spark lit les transactions depuis Kafka et :

* Parse les messages JSON
* Applique du feature engineering
* Charge un modèle ML entraîné
* Calcule un score de risque
* Classe la transaction comme frauduleuse ou non

---

### 3️⃣ Stockage des Données

Les résultats sont enregistrés dans PostgreSQL :

* `transactions_enriched`
* `fraud_alerts`

---

### 4️⃣ Visualisation & Monitoring

Le dashboard Streamlit permet de visualiser :

* Le taux de fraude
* Les transactions à haut risque
* La distribution des scores
* L’activité récente

---

# 🚀 Lancer le Projet (Étapes)

## 1️⃣ Démarrer l’infrastructure Docker

```bash
docker compose up -d
```

Vérifier que les conteneurs tournent :

```bash
docker ps
```

---

## 2️⃣ Créer le Topic Kafka

```bash
docker exec -it realtime-fraud-pipeline-kafka-1 bash -lc \
"kafka-topics --bootstrap-server kafka:29092 --create \
--topic transactions --partitions 1 --replication-factor 1 || true"
```

---

## 3️⃣ Entraîner le Modèle ML

```bash
docker exec -it realtime-fraud-pipeline-spark-1 bash -lc \
"/opt/spark/bin/spark-submit --master local[*] /opt/streaming/train_model.py"
```

Cela génère le modèle dans :

```
/opt/streaming/model
```

---

## 4️⃣ Lancer le Streaming

```bash
docker exec -it realtime-fraud-pipeline-spark-1 bash -lc \
"/opt/spark/bin/spark-submit --master local[*] /opt/streaming/spark_job.py"
```

Laisser ce terminal ouvert.

---

## 5️⃣ Lancer le Producteur

```bash
python src/producer/producer.py
```

---

## 6️⃣ Ouvrir le Dashboard

```
http://localhost:8501
```

---

# 📊 Prise de Décision dans un Cadre Métier

Ce projet permet de soutenir des décisions opérationnelles en temps réel dans un environnement financier.

---

## 🔎 1. Blocage des Transactions à Risque

Si :

```
risk_score > seuil
```

Alors :

* Blocage automatique de la transaction
* Mise en revue manuelle
* Déclenchement d’une alerte

---

## 📈 2. Surveillance Dynamique du Risque

Le dashboard permet d’identifier :

* L’évolution du taux de fraude
* Les commerçants à risque
* Les pays suspects
* Les clients à forte exposition

Cela permet :

* De prioriser les enquêtes
* D’ajuster les seuils de risque
* D’optimiser les ressources antifraude

---

## 💰 3. Optimisation Financière

La fraude implique un arbitrage :

| Trop strict                  | Trop permissif       |
| ---------------------------- | -------------------- |
| Blocage de clients légitimes | Pertes financières   |
| Insatisfaction client        | Risque réglementaire |

Les équipes métiers peuvent :

* Ajuster le seuil de fraude
* Optimiser précision vs rappel
* Minimiser le coût des faux positifs

---

# 🧠 Modèle Machine Learning

Modèle utilisé :

* **Régression Logistique (Spark MLlib)**

Variables utilisées :

* Montant
* Pays
* Commerçant
* Catégorie
* Encodage des variables catégorielles

Sortie :

```
risk_score ∈ [0,1]
```

---

# 🔄 Améliorations Futures

* Détection de dérive du modèle (data drift)
* Monitoring avancé
* Microservices pour le scoring
* Remplacement par :

  * Gradient Boosted Trees
  * XGBoost
  * Deep Learning
* Déploiement Cloud (AWS / GCP)
* Gestion des erreurs via Kafka Dead Letter Queue

---

# 📌 Ce que Démontre ce Projet

* Un pipeline temps réel complet
* L’industrialisation d’un modèle ML
* Une architecture orientée production
* Une logique décisionnelle métier
* Une approche Data Engineering avancée

