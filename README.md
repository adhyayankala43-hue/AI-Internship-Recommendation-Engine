# AI-Powered Multi-Tier Internship Recommendation and Visual Analytics Engine

![Domain](https://img.shields.io/badge/Domain-AI%20%26%20Data%20Science-blue)
![Python](https://img.shields.io/badge/Backend-Python%20Flask-green)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c)
![Frontend](https://img.shields.io/badge/Frontend-Tailwind%20CSS%20%7C%20Chart.js-38B2AC)

## 📌 Project Overview
The **AI Internship Recommendation Engine** is an advanced, multi-tier web application designed to bridge the gap between academic profile data and corporate internship opportunities. By integrating traditional Natural Language Processing (NLP), collaborative filtering, and deep learning architectures into a unified platform, the system delivers precise, personalized, and multi-perspective recommendations to students.

## 🎯 Key Objectives & Features
* **Precision Matching:** Deploys content-based vectorization (TF-IDF) to align student skills and academic degrees directly with corporate requirements.
* **Crowdsourced Wisdom Integration:** Leverages collaborative filtering and peer interaction patterns to surface popular or high-demand opportunities.
* **Neural Network Generalization:** Utilizes a PyTorch Multi-Layer Perceptron (MLP) deep learning model to uncover latent interactions and predict suitability scores.
* **Interactive Visualization:** Provides real-time, interactive UI feedback via Chart.js radar (spider) graphs mapping company demands against user profiles.
* **Geographical Flexibility:** Enforces hard academic filters (CGPA cutoffs) alongside smart location overrides to prioritize local or remote positions.

## 🛠️ Technology Stack
The architecture follows a modern Client-Server (Decoupled API & View) pattern.

**Backend & Data Processing**
* **Framework:** Python Flask (RESTful routing, JSON serialization)
* **Data Engineering:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-learn (TfidfVectorizer, cosine_similarity, MinMaxScaler)
* **Deep Learning:** PyTorch (`nn.Module`, `nn.Embedding`, `nn.Linear`)

**Frontend Interface**
* **Markup & Styling:** HTML5, Tailwind CSS (via CDN)
* **Data Visualization:** Chart.js (Interactive Radar/Spider Graphs)

**Persistence Layer**
* Optimized in-memory Pandas DataFrames merged from structured flat-file CSVs.

## 🏗️ System Architecture

```text
+-------------------------------------------------------------+
|                        Client Layer                         |
|        HTML5 / Tailwind CSS / JavaScript (Chart.js)         |
+-------------------------------------------------------------+
                               |
                      HTTP POST (/recommend)
                               v
 +-------------------------------------------------------------+
 |                      Application Layer                      |
 |                   Python Flask Web Server                   |
 +-------------------------------------------------------------+
               |               |               |
               v               v               v
       +--------------+ +-------------+ +-------------+
       | Pandas/NumPy | | Scikit-learn| | PyTorch(MLP)|
       | Data Engine  | | TF-IDF & Cos| | Neural Net  |
       +--------------+ +-------------+ +-------------+
               |               |               |
               +-------+-------+-------+-------+
                               |
                               v
+-------------------------------------------------------------+
|                      Persistence Layer                      |
|    Flat-File CSVs (Internships, Ratings, Interactions)      |
+-------------------------------------------------------------+
