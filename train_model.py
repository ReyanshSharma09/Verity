"""
Script to train and save the fraud detection model
Run this once before starting the server
"""

from data_simulator import TransactionSimulator
from fraud_model import FraudDetectionModel

def main():
    print("=" * 60)
    print("🚀 FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n📊 Step 1: Generating synthetic training data...")
    simulator = TransactionSimulator()
    transactions = simulator.generate_dataset(n_transactions=10000, fraud_ratio=0.15)
    df = simulator.to_dataframe(transactions)
    
    print(f"   ✅ Generated {len(df)} transactions")
    print(f"   ✅ Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"   ✅ Legitimate cases: {len(df) - df['is_fraud'].sum()}")
    
    # Step 2: Train model
    print("\n🧠 Step 2: Training ML model...")
    model = FraudDetectionModel()
    metrics = model.train(df)
    
    print(f"\n   ✅ Model accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   ✅ ROC-AUC score: {metrics['roc_auc']:.4f}")
    
    # Step 3: Save model
    print("\n💾 Step 3: Saving model...")
    model.save("model.pkl")
    
    # Step 4: Verify saved model
    print("\n🔍 Step 4: Verifying saved model...")
    test_model = FraudDetectionModel()
    test_model.load("model.pkl")
    
    # Test prediction
    test_txn = simulator.generate_transaction(force_fraud=True)
    result = test_model.predict(test_txn)
    print(f"   ✅ Test prediction successful")
    print(f"   ✅ Test transaction risk score: {result['risk_score']}/100")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Run 'python app.py' to start the server")
    print("   2. Open 'index.html' in your browser")
    print("   3. The app will now use real ML predictions!")

if __name__ == "__main__":
    main()
