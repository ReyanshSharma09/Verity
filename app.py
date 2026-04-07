"""
Flask API Server for Fraud Detection
Connects ML model with frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from data_simulator import TransactionSimulator
from fraud_model import FraudDetectionModel
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
simulator = TransactionSimulator()
model = FraudDetectionModel()

# Load or train model
if os.path.exists("model.pkl"):
    model.load("model.pkl")
    print("✅ Loaded existing model")
else:
    print("⚠️ No saved model found. Training new model...")
    transactions = simulator.generate_dataset(5000)
    df = simulator.to_dataframe(transactions)
    model.train(df)
    model.save("model.pkl")
    print("✅ New model trained and saved")

# Store transactions in memory (in production, use a database)
transaction_history = []
pending_alerts = []

def format_transaction_for_frontend(txn, prediction):
    """Format transaction data for the frontend"""
    return {
        "id": txn["id"],
        "merchant": txn["merchant"],
        "category": txn["merchant_category"],
        "icon": txn["icon"],
        "amount": txn["amount"],
        "location": txn["location"],
        "distance": txn["distance_from_home"],
        "timestamp": txn["timestamp"],
        "hour": txn["hour"],
        
        # ML Prediction
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "risk_color": prediction["risk_color"],
        "is_fraud_prediction": prediction["is_fraud"],
        "fraud_probability": prediction["fraud_probability"],
        "risk_factors": prediction["risk_factors"],
        
        # Status
        "status": "pending" if prediction["risk_score"] >= 50 else "verified",
        "reviewed": False
    }


@app.route("/")
def home():
    """API Home"""
    return jsonify({
        "name": "SafePay Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "GET /api/transactions - Get all transactions",
            "GET /api/transactions/generate - Generate new transaction",
            "GET /api/alerts - Get pending alerts",
            "POST /api/transaction/approve - Approve a transaction",
            "POST /api/transaction/block - Block a transaction",
            "GET /api/stats - Get account statistics"
        ]
    })


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Get all transactions"""
    return jsonify({
        "success": True,
        "count": len(transaction_history),
        "transactions": transaction_history
    })


@app.route("/api/transactions/generate", methods=["GET"])
def generate_transaction():
    """Generate a new random transaction with ML prediction"""
    
    # Check if we should force a suspicious transaction
    force_fraud = request.args.get("fraud", "").lower() == "true"
    
    # Generate transaction
    txn = simulator.generate_transaction(force_fraud=force_fraud if force_fraud else None)
    
    # Get ML prediction
    prediction = model.predict(txn)
    
    # Format for frontend
    formatted_txn = format_transaction_for_frontend(txn, prediction)
    
    # Add to history
    transaction_history.insert(0, formatted_txn)
    
    # If high risk, add to alerts
    if prediction["risk_score"] >= 50:
        pending_alerts.insert(0, formatted_txn)
    
    # Keep only last 50 transactions
    if len(transaction_history) > 50:
        transaction_history.pop()
    
    return jsonify({
        "success": True,
        "transaction": formatted_txn,
        "is_alert": prediction["risk_score"] >= 50
    })


@app.route("/api/transactions/batch", methods=["GET"])
def generate_batch():
    """Generate multiple transactions at once"""
    count = min(int(request.args.get("count", 10)), 20)
    
    transactions = []
    for _ in range(count):
        txn = simulator.generate_transaction()
        prediction = model.predict(txn)
        formatted_txn = format_transaction_for_frontend(txn, prediction)
        transactions.append(formatted_txn)
        transaction_history.insert(0, formatted_txn)
        
        if prediction["risk_score"] >= 50:
            pending_alerts.insert(0, formatted_txn)
    
    return jsonify({
        "success": True,
        "count": len(transactions),
        "transactions": transactions
    })


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Get pending fraud alerts"""
    # Filter only unreviewed high-risk transactions
    active_alerts = [a for a in pending_alerts if not a.get("reviewed", False)]
    
    return jsonify({
        "success": True,
        "count": len(active_alerts),
        "alerts": active_alerts
    })


@app.route("/api/transaction/approve", methods=["POST"])
def approve_transaction():
    """Approve a flagged transaction"""
    data = request.json
    txn_id = data.get("id")
    
    # Find and update transaction
    for txn in transaction_history:
        if txn["id"] == txn_id:
            txn["status"] = "approved"
            txn["reviewed"] = True
            txn["reviewed_at"] = datetime.now().isoformat()
            break
    
    # Remove from alerts
    for i, alert in enumerate(pending_alerts):
        if alert["id"] == txn_id:
            pending_alerts[i]["reviewed"] = True
            break
    
    return jsonify({
        "success": True,
        "message": f"Transaction {txn_id} approved",
        "action": "approved"
    })


@app.route("/api/transaction/block", methods=["POST"])
def block_transaction():
    """Block a fraudulent transaction"""
    data = request.json
    txn_id = data.get("id")
    report_fraud = data.get("report_fraud", True)
    
    # Find and update transaction
    for txn in transaction_history:
        if txn["id"] == txn_id:
            txn["status"] = "blocked"
            txn["reviewed"] = True
            txn["reviewed_at"] = datetime.now().isoformat()
            txn["fraud_reported"] = report_fraud
            break
    
    # Remove from alerts
    for i, alert in enumerate(pending_alerts):
        if alert["id"] == txn_id:
            pending_alerts[i]["reviewed"] = True
            break
    
    return jsonify({
        "success": True,
        "message": f"Transaction {txn_id} blocked",
        "action": "blocked",
        "fraud_reported": report_fraud
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get account statistics"""
    total = len(transaction_history)
    blocked = len([t for t in transaction_history if t.get("status") == "blocked"])
    approved = len([t for t in transaction_history if t.get("status") == "approved"])
    pending = len([t for t in transaction_history if t.get("status") == "pending"])
    verified = len([t for t in transaction_history if t.get("status") == "verified"])
    
    # Calculate total amounts
    total_amount = sum(t["amount"] for t in transaction_history)
    blocked_amount = sum(t["amount"] for t in transaction_history if t.get("status") == "blocked")
    
    return jsonify({
        "success": True,
        "stats": {
            "total_transactions": total,
            "blocked": blocked,
            "approved": approved,
            "pending": pending,
            "verified": verified,
            "total_amount": round(total_amount, 2),
            "blocked_amount": round(blocked_amount, 2),
            "fraud_prevention_rate": round((blocked / total * 100) if total > 0 else 0, 1),
            "pending_alerts": len([a for a in pending_alerts if not a.get("reviewed", False)])
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_transaction():
    """Analyze a custom transaction"""
    data = request.json
    
    # Create transaction from request data
    txn = {
        "features": {
            "amount": data.get("amount", 0),
            "amount_ratio": data.get("amount", 0) / 45.0,  # Assuming avg is $45
            "distance": data.get("distance", 0),
            "hour": data.get("hour", 12),
            "is_unusual_hour": 1 if data.get("hour", 12) < 6 or data.get("hour", 12) > 22 else 0,
            "is_unusual_category": data.get("unusual_category", 0),
            "is_high_risk_merchant": data.get("high_risk", 0),
            "is_weekend": data.get("weekend", 0)
        }
    }
    
    prediction = model.predict(txn)
    
    return jsonify({
        "success": True,
        "analysis": prediction
    })


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """Get information about the ML model"""
    return jsonify({
        "success": True,
        "model": {
            "type": "Random Forest Classifier",
            "features": model.feature_names,
            "is_trained": model.is_trained,
            "risk_thresholds": {
                "low": "0-39",
                "medium": "40-69", 
                "high": "70-100"
            }
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 SAFEPAY FRAUD DETECTION API")
    print("=" * 50)
    print(f"✅ Model loaded: {model.is_trained}")
    print(f"📍 Server starting at http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, port=5000)
