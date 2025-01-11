# Environmental Monitoring and Pollution Prediction System

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)

---

## Overview

This project implements a robust pipeline for real-time **environmental monitoring**, **pollution prediction**, and **system monitoring** using modern **MLOps tools**. The project includes:
- Automated **data collection** using OpenWeatherMap API.
- A predictive model trained on environmental data using an **LSTM architecture**.
- Deployment of the model with a **Flask API**.
- System performance monitoring with **Prometheus** and **Grafana**.

---

## Features

1. **Task 1**: Data collection and management using DVC.
2. **Task 2**: Pollution trend prediction using MLflow for model tracking.
3. **Task 3**: Real-time monitoring with Prometheus and Grafana.

---

## Technologies Used

- **Python**: For scripting and model development.
- **DVC**: For data version control.
- **MLflow**: For experiment tracking and model deployment.
- **Flask**: To expose the trained model as an API.
- **Prometheus**: To monitor system metrics.
- **Grafana**: To visualize metrics in real-time.
- **TensorFlow/Keras**: To build the LSTM model.
- **OpenWeatherMap API**: To fetch real-time environmental data.

---

## Project Structure

```plaintext
.
├── Data/                          # Folder to store collected data
│   └── environmental_data.csv     # Collected environmental data
├── models/                        # Folder to store trained models
│   └── best_lstm_model.h5         # Best trained LSTM model
├── app.py                         # Flask API for serving predictions
├── data_fetch.py                  # Script to fetch real-time environmental data
├── data_preparation.py            # Data cleaning and preparation script
├── lstm_training.py               # Script to train LSTM model and log metrics
├── prometheus.yml                 # Prometheus configuration file
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── dashboard.json                 # Grafana dashboard configuration