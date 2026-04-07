"""
Fraud Detection ML Model
Uses Random Forest with multiple features for fraud prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'amount', 'amount_ratio', 'distance', 'hour',
            'is_unusual_hour', 'is_unusual_category', 
            'is_high_risk_merchant', 'is_weekend'
        ]
        self.is_trained = False
        
    def prepare_features(self, transaction):
        """Extract features from a transaction dict"""
        features = transaction.get("features", transaction)
        
        X = np.array([[
            features.get("amount", 0),
            features.get("amount_ratio", 1),
            features.get("distance", 0),
            features.get("hour", 12),
            features.get("is_unusual_hour", 0),
            features.get("is_unusual_category", 0),
            features.get("is_high_risk_merchant", 0),
            features.get("is_weekend", 0)
        ]])
        
        return X
    
    def train(self, df):
        """Train the fraud detection model"""
        print("🔄 Training fraud detection model...")
        
        # Prepare features and target
        X = df[self.feature_names].values
        y = df['is_fraud'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n📊 Model Performance:")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        
        # Feature importance
        print("\n🎯 Feature Importance:")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"  {row['feature']:25} {bar} {row['importance']:.3f}")
        
        return {
            "accuracy": (y_pred == y_test).mean(),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
    
    def predict(self, transaction):
        """
        Predict fraud probability for a single transaction
        Returns: dict with prediction details
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        # Prepare features
        X = self.prepare_features(transaction)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        fraud_probability = probabilities[1]
        risk_score = int(fraud_probability * 100)
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "HIGH"
            risk_color = "danger"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            risk_color = "warning"
        else:
            risk_level = "LOW"
            risk_color = "success"
        
        # Generate risk factors explanation
        risk_factors = self._explain_prediction(transaction, fraud_probability)
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(fraud_probability, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_factors": risk_factors
        }
    
    def _explain_prediction(self, transaction, fraud_prob):
        """Generate human-readable explanation for the prediction"""
        features = transaction.get("features", transaction)
        factors = []
        
        # Amount analysis
        amount = features.get("amount", 0)
        amount_ratio = features.get("amount_ratio", 1)
        if amount_ratio > 10:
            factors.append({
                "factor": "Unusually high amount",
                "detail": f"${amount:.2f} is {amount_ratio:.1f}x your average",
                "severity": "high",
                "icon": "fas fa-dollar-sign"
            })
        elif amount_ratio > 5:
            factors.append({
                "factor": "Higher than usual amount",
                "detail": f"${amount:.2f} is {amount_ratio:.1f}x your average",
                "severity": "medium",
                "icon": "fas fa-dollar-sign"
            })
        
        # Distance analysis
        distance = features.get("distance", 0)
        if distance > 1000:
            factors.append({
                "factor": "Very far from home",
                "detail": f"{distance:.0f} miles from your location",
                "severity": "high",
                "icon": "fas fa-map-marker-alt"
            })
        elif distance > 100:
            factors.append({
                "factor": "Unusual location",
                "detail": f"{distance:.0f} miles from your location",
                "severity": "medium",
                "icon": "fas fa-map-marker-alt"
            })
        
        # Time analysis
        if features.get("is_unusual_hour", 0):
            factors.append({
                "factor": "Unusual time",
                "detail": f"Transaction at {features.get('hour', 0)}:00 is outside normal hours",
                "severity": "medium",
                "icon": "fas fa-clock"
            })
        
        # Merchant analysis
        if features.get("is_high_risk_merchant", 0):
            factors.append({
                "factor": "High-risk merchant",
                "detail": "This merchant category has elevated fraud risk",
                "severity": "high",
                "icon": "fas fa-store"
            })
        
        if features.get("is_unusual_category", 0):
            factors.append({
                "factor": "New category",
                "detail": "You don't usually make purchases in this category",
                "severity": "low",
                "icon": "fas fa-tag"
            })
        
        # Weekend analysis
        if features.get("is_weekend", 0) and fraud_prob > 0.5:
            factors.append({
                "factor": "Weekend transaction",
                "detail": "Combined with other factors, this increases risk",
                "severity": "low",
                "icon": "fas fa-calendar"
            })
        
        # If no risk factors but still flagged
        if not factors and fraud_prob > 0.4:
            factors.append({
                "factor": "Pattern anomaly",
                "detail": "This transaction doesn't match your typical behavior",
                "severity": "medium",
                "icon": "fas fa-chart-line"
            })
        
        return factors
    
    def save(self, filepath="model.pkl"):
        """Save model and scaler"""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath="model.pkl"):
        """Load model and scaler"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.is_trained = data["is_trained"]
            print(f"✅ Model loaded from {filepath}")
            return True
        return False


# Test the model
if __name__ == "__main__":
    from data_simulator import TransactionSimulator
    
    print("=" * 50)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Generate training data
    simulator = TransactionSimulator()
    print("\n📊 Generating training data...")
    transactions = simulator.generate_dataset(5000, fraud_ratio=0.15)
    df = simulator.to_dataframe(transactions)
    
    # Train model
    model = FraudDetectionModel()
    metrics = model.train(df)
    
    # Save model
    model.save("model.pkl")
    
    # Test predictions
    print("\n" + "=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)
    
    # Test with new transactions
    for i in range(5):
        txn = simulator.generate_transaction()
        result = model.predict(txn)
        
        print(f"\n{'🔴' if result['is_fraud'] else '🟢'} {txn['merchant']}")
        print(f"   Amount: ${txn['amount']:.2f}")
        print(f"   Risk Score: {result['risk_score']}/100 ({result['risk_level']})")
        print(f"   Actual: {'FRAUD' if txn['is_fraud'] else 'LEGITIMATE'}")
        
        if result['risk_factors']:
            print("   Risk Factors:")
            for factor in result['risk_factors']:
                print(f"      - {factor['factor']}: {factor['detail']}")
