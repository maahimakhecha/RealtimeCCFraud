﻿# Real-time Credit Card Fraud Detection

A novel framework that leverages Apache Kafka for real-time data streaming and Q-learning, a reinforcement learning algorithm, to enhance the efficiency and accuracy of fraud detection. Our system achieves real-time data ingestion and processing, enabling rapid decision-making to combat potential fraud. Through Q-learning, our system continuously learns and adapts its decision-making strategies based on feedback, improving its accuracy over time. In extensive evaluations, our model achieved an accuracy rate exceeding 95%.

## Features

- **Q-Learning Framework**: Advanced reinforcement learning approach using Q-learning for adaptive fraud detection
- **Real-time Kafka Streaming**: Apache Kafka integration for seamless real-time data ingestion and processing
- **Dual ML Approach**: Combines supervised learning and reinforcement learning for robust fraud detection
- **High Accuracy**: Achieves over 95% accuracy in fraud detection
- **Advanced Models**: 
  - Supervised Learning: Traditional ML models for fraud classification
  - Q-Learning RL: Adaptive agent that learns optimal fraud detection strategies
- **Comprehensive Evaluation**: ROC curves, training metrics, and performance visualizations
  
## Project Structure

```
RealtimeCCFraud/
├── data/                    # Data storage
│   ├── raw/                # Raw transaction data
│   └── processed/          # Preprocessed data
├── models/                 # Trained models and visualizations
│   ├── supervised_model.pth
│   ├── rl_model.pth
│   ├── rl_model_kafka.pth
│   └── *.png              # Training plots and metrics
├── notebooks/              # Jupyter notebooks
│   ├── 1_EDA.ipynb        # Exploratory Data Analysis
│   └── 2_supervised_evaluation.ipynb
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── supervised_model.py
│   ├── rl_agent.py
│   ├── rl_environment.py
│   ├── utils.py
│   └── streaming/          # Kafka integration
│       ├── kafka_producer.py
│       └── kafka_consumer.py
├── requirements.txt
└── README.md
```

## Key Components

### Supervised Learning
- Traditional machine learning models for fraud classification
- Feature engineering and model evaluation
- Performance metrics and visualizations

### Q-Learning Reinforcement Learning
- Novel Q-learning framework for adaptive fraud detection
- Continuous learning and strategy adaptation based on feedback
- Real-time decision making with over 95% accuracy
- Environment simulation for optimal policy learning

### Kafka Streaming
- Producer: Sends transaction data to Kafka topics
- Consumer: Processes transactions in real-time
- Integration with ML models for live predictions

## Model Performance

Our novel Q-learning framework achieves exceptional performance:
- **95%+ Accuracy**: Outstanding fraud detection accuracy
- **Real-time Processing**: Sub-second response times for fraud detection
- **Continuous Learning**: Adaptive improvement over time
- **Comprehensive Evaluation**: 
  - ROC curves and AUC scores
  - Training loss plots
  - Q-learning episode scores and convergence
  - Performance comparisons between supervised and RL approaches
