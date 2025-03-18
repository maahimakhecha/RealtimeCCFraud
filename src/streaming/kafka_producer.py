import os
import json
import pandas as pd
from kafka import KafkaProducer

def load_transactions(file_path, n_samples=100):
    df = pd.read_csv(file_path)
    return df.sample(n=n_samples)

def main():
    processed_data_path = os.path.join('data', 'processed', 'test.csv')
    transactions = load_transactions(processed_data_path, n_samples=100)
    
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print("Kafka Producer is ready.")
    print("Press Enter to send each transaction. To exit, press Ctrl+C.\n")
    
    for idx, (_, row) in enumerate(transactions.iterrows(), 1):
        input(f"Press Enter to send message {idx}...")
        message = row.to_dict()
        producer.send('transactions', message)
        producer.flush()
        print(f"Sent message {idx}: {message}\n")
    
    print("All messages sent.")

if __name__ == '__main__':
    main()
