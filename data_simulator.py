"""
Transaction Data Simulator for Fraud Detection
Generates realistic transaction data with fraud patterns
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class TransactionSimulator:
    def __init__(self):
        # Normal merchants (low risk)
        self.normal_merchants = [
            {"name": "Amazon", "category": "shopping", "icon": "fab fa-amazon"},
            {"name": "Starbucks", "category": "food", "icon": "fas fa-coffee"},
            {"name": "Walmart", "category": "grocery", "icon": "fas fa-shopping-cart"},
            {"name": "Netflix", "category": "entertainment", "icon": "fab fa-netflix"},
            {"name": "Spotify", "category": "entertainment", "icon": "fab fa-spotify"},
            {"name": "Shell Gas", "category": "gas", "icon": "fas fa-gas-pump"},
            {"name": "Chipotle", "category": "food", "icon": "fas fa-utensils"},
            {"name": "Target", "category": "shopping", "icon": "fas fa-bullseye"},
            {"name": "CVS Pharmacy", "category": "pharmacy", "icon": "fas fa-prescription-bottle"},
            {"name": "Uber", "category": "transport", "icon": "fab fa-uber"},
        ]
        
        # Suspicious merchants (high risk)
        self.suspicious_merchants = [
            {"name": "Las Vegas Casino", "category": "gambling", "icon": "fas fa-dice"},
            {"name": "Crypto Exchange XYZ", "category": "crypto", "icon": "fab fa-bitcoin"},
            {"name": "Foreign ATM Withdrawal", "category": "atm", "icon": "fas fa-money-bill"},
            {"name": "Online Betting Site", "category": "gambling", "icon": "fas fa-futbol"},
            {"name": "Wire Transfer Intl", "category": "transfer", "icon": "fas fa-exchange-alt"},
            {"name": "Unknown Merchant", "category": "unknown", "icon": "fas fa-question"},
            {"name": "Adult Services LLC", "category": "adult", "icon": "fas fa-ban"},
            {"name": "Luxury Jewelry Store", "category": "luxury", "icon": "fas fa-gem"},
        ]
        
        # User's home location (for distance calculation)
        self.home_location = {"city": "New York", "lat": 40.7128, "lon": -74.0060}
        
        # Normal locations
        self.normal_locations = [
            {"city": "New York, NY", "lat": 40.7128, "lon": -74.0060},
            {"city": "Brooklyn, NY", "lat": 40.6782, "lon": -73.9442},
            {"city": "Jersey City, NJ", "lat": 40.7178, "lon": -74.0431},
        ]
        
        # Suspicious locations (far from home)
        self.suspicious_locations = [
            {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398},
            {"city": "Miami, FL", "lat": 25.7617, "lon": -80.1918},
            {"city": "Moscow, Russia", "lat": 55.7558, "lon": 37.6173},
            {"city": "Lagos, Nigeria", "lat": 6.5244, "lon": 3.3792},
            {"city": "Bangkok, Thailand", "lat": 13.7563, "lon": 100.5018},
        ]
        
        # User spending patterns (for baseline)
        self.user_profile = {
            "avg_transaction": 45.00,
            "max_normal_transaction": 200.00,
            "typical_categories": ["shopping", "food", "gas", "entertainment"],
            "typical_hours": list(range(7, 23)),  # 7 AM to 11 PM
        }
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in miles"""
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        miles = 3956 * c  # Earth's radius in miles
        return round(miles, 2)
    
    def generate_transaction(self, force_fraud=None):
        """
        Generate a single transaction
        force_fraud: None (random), True (force fraud), False (force legitimate)
        """
        
        # Determine if this transaction is fraudulent
        if force_fraud is None:
            is_fraud = random.random() < 0.15  # 15% fraud rate
        else:
            is_fraud = force_fraud
        
        # Generate transaction ID
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        
        # Generate timestamp
        if is_fraud and random.random() < 0.4:
            # Fraud often happens at unusual hours
            hour = random.choice([0, 1, 2, 3, 4, 5, 23])
        else:
            hour = random.choice(self.user_profile["typical_hours"])
        
        days_ago = random.randint(0, 7)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        timestamp = timestamp.replace(hour=hour)
        
        # Select merchant and location
        if is_fraud:
            merchant = random.choice(self.suspicious_merchants)
            location = random.choice(self.suspicious_locations)
            
            # Fraudulent amounts are often unusual
            amount_type = random.choice(["very_high", "round_number", "just_below_limit"])
            if amount_type == "very_high":
                amount = round(random.uniform(500, 5000), 2)
            elif amount_type == "round_number":
                amount = float(random.choice([500, 1000, 2000, 2500, 3000]))
            else:
                amount = round(random.uniform(450, 499), 2)  # Just below $500 reporting limit
        else:
            merchant = random.choice(self.normal_merchants)
            location = random.choice(self.normal_locations)
            amount = round(random.uniform(5, self.user_profile["max_normal_transaction"]), 2)
        
        # Calculate distance from home
        distance = self.calculate_distance(
            self.home_location["lat"], self.home_location["lon"],
            location["lat"], location["lon"]
        )
        
        # Calculate feature values for ML
        amount_ratio = amount / self.user_profile["avg_transaction"]
        is_unusual_hour = 1 if hour < 6 or hour > 22 else 0
        is_unusual_category = 0 if merchant["category"] in self.user_profile["typical_categories"] else 1
        is_high_risk_merchant = 1 if merchant in self.suspicious_merchants else 0
        
        transaction = {
            "id": transaction_id,
            "merchant": merchant["name"],
            "merchant_category": merchant["category"],
            "icon": merchant["icon"],
            "amount": amount,
            "location": location["city"],
            "distance_from_home": distance,
            "timestamp": timestamp.isoformat(),
            "hour": hour,
            "day_of_week": timestamp.weekday(),
            
            # ML Features
            "features": {
                "amount": amount,
                "amount_ratio": round(amount_ratio, 2),
                "distance": distance,
                "hour": hour,
                "is_unusual_hour": is_unusual_hour,
                "is_unusual_category": is_unusual_category,
                "is_high_risk_merchant": is_high_risk_merchant,
                "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
            },
            
            # Ground truth (for training)
            "is_fraud": is_fraud
        }
        
        return transaction
    
    def generate_dataset(self, n_transactions=1000, fraud_ratio=0.15):
        """Generate a dataset for training"""
        transactions = []
        
        n_fraud = int(n_transactions * fraud_ratio)
        n_legitimate = n_transactions - n_fraud
        
        # Generate fraudulent transactions
        for _ in range(n_fraud):
            transactions.append(self.generate_transaction(force_fraud=True))
        
        # Generate legitimate transactions
        for _ in range(n_legitimate):
            transactions.append(self.generate_transaction(force_fraud=False))
        
        random.shuffle(transactions)
        return transactions
    
    def generate_realtime_transaction(self):
        """Generate a single real-time transaction for demo"""
        return self.generate_transaction()
    
    def to_dataframe(self, transactions):
        """Convert transactions to pandas DataFrame for ML"""
        data = []
        for t in transactions:
            row = {
                "id": t["id"],
                "amount": t["features"]["amount"],
                "amount_ratio": t["features"]["amount_ratio"],
                "distance": t["features"]["distance"],
                "hour": t["features"]["hour"],
                "is_unusual_hour": t["features"]["is_unusual_hour"],
                "is_unusual_category": t["features"]["is_unusual_category"],
                "is_high_risk_merchant": t["features"]["is_high_risk_merchant"],
                "is_weekend": t["features"]["is_weekend"],
                "is_fraud": 1 if t["is_fraud"] else 0
            }
            data.append(row)
        
        return pd.DataFrame(data)


# Test the simulator
if __name__ == "__main__":
    simulator = TransactionSimulator()
    
    print("=" * 50)
    print("TRANSACTION DATA SIMULATOR")
    print("=" * 50)
    
    # Generate sample transactions
    print("\n📊 Generating sample transactions...\n")
    
    for i in range(5):
        txn = simulator.generate_transaction()
        fraud_status = "🔴 FRAUD" if txn["is_fraud"] else "🟢 LEGITIMATE"
        print(f"{fraud_status}")
        print(f"   Merchant: {txn['merchant']}")
        print(f"   Amount: ${txn['amount']}")
        print(f"   Location: {txn['location']}")
        print(f"   Distance: {txn['distance_from_home']} miles")
        print()
    
    # Generate dataset
    print("\n📈 Generating training dataset...")
    dataset = simulator.generate_dataset(1000)
    df = simulator.to_dataframe(dataset)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    print(f"Legitimate cases: {len(df) - df['is_fraud'].sum()}")
    
    # Save dataset
    df.to_csv("transaction_data.csv", index=False)
    print("\n✅ Dataset saved to transaction_data.csv")
