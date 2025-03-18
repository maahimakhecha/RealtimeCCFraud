# src/streaming/kafka_consumer.py

import os
import json
from kafka import KafkaConsumer
from src.rl_environment import FraudDetectionEnv


def main():
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        group_id='fraud_detection_consumer',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    env = FraudDetectionEnv(feature_dim=30, pretrained_model_path=None)
    print("Starting Kafka Consumer. Listening for transactions...")
    for message in consumer:
        transaction = message.value
        print("\nReceived transaction:")
        print(transaction)
        state = env.set_transaction(transaction)
        action = env.action_space.sample()  # For demo, use random action (or integrate the RL agent)
        print(f"Selected action: {action}")
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Info: {info}")
        env.reset()

if __name__ == '__main__':
    main()
