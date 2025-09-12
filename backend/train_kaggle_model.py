#!/usr/bin/env python3
"""
Train model with Kaggle dataset (heart, tulip, swan, rosetta)
"""

import os
from transfer_learning_model import TransferLearningLatteArtClassifier

class KaggleLatteArtClassifier(TransferLearningLatteArtClassifier):
    """Extended classifier for Kaggle dataset with rosetta category"""
    
    def __init__(self, img_size=(224, 224)):
        # Update class names for Kaggle dataset
        self.class_names = ['heart', 'tulip', 'swan', 'rosetta']
        self.img_size = img_size
        self.model = None
        self.load_model('kaggle_latte_art_model.h5')

def train_kaggle_model():
    """Train model with Kaggle dataset"""
    print("🚀 Training Model with Kaggle Dataset")
    print("=" * 40)
    
    # Check if Kaggle dataset exists
    if not os.path.exists("dataset_kaggle_augmented"):
        print("❌ Kaggle augmented dataset not found!")
        print("💡 Run: python3 kaggle_dataset_manager.py first")
        return False
    
    # Initialize classifier with new categories
    classifier = KaggleLatteArtClassifier()
    
    # Create new model
    print("🏗️  Creating model for Kaggle categories...")
    classifier.create_transfer_model()
    
    # Train with enhanced settings
    print("🎓 Training with Kaggle dataset...")
    print("   • Categories: heart, tulip, swan, rosetta")
    print("   • More epochs: 100")
    print("   • Batch size: 16")
    print("   • Large dataset: 460+ images")
    
    try:
        # Initial training
        history = classifier.train(
            data_dir="dataset_kaggle_augmented",
            epochs=100,
            batch_size=16,
            validation_split=0.2
        )
        
        # Fine-tune
        print("\n🔧 Fine-tuning the model...")
        fine_tune_history = classifier.fine_tune(
            data_dir="dataset_kaggle_augmented",
            epochs=25,
            batch_size=16,
            validation_split=0.2
        )
        
        # Save the model
        classifier.save_model("kaggle_latte_art_model.h5")
        
        print("✅ Kaggle model training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_kaggle_model():
    """Test the trained Kaggle model"""
    print("\n🧪 Testing Kaggle Model Performance")
    print("=" * 40)
    
    classifier = KaggleLatteArtClassifier()
    classifier.load_model("kaggle_latte_art_model.h5")
    
    if classifier.model is None:
        print("❌ Model failed to load!")
        return
    
    # Test each category
    categories = ['heart', 'tulip', 'swan', 'rosetta']
    
    for category in categories:
        category_dir = os.path.join("dataset_kaggle", category)
        if os.path.exists(category_dir):
            # Test first few images
            images = [f for f in os.listdir(category_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            
            print(f"\n📸 Testing {category} images:")
            correct = 0
            for image_file in images:
                image_path = os.path.join(category_dir, image_file)
                try:
                    art_type, confidence = classifier.predict(image_path)
                    is_correct = art_type.lower() == category.lower()
                    if is_correct:
                        correct += 1
                    
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} {image_file} → {art_type} ({confidence:.3f})")
                except Exception as e:
                    print(f"   ❌ {image_file} → ERROR: {e}")
            
            accuracy = (correct / len(images)) * 100 if images else 0
            print(f"   📊 Accuracy: {accuracy:.1f}% ({correct}/{len(images)})")

if __name__ == "__main__":
    success = train_kaggle_model()
    if success:
        test_kaggle_model()
        print("\n🎉 Kaggle model training complete!")
        print("💡 This model should have much better performance!")
        print("📁 Model saved as: kaggle_latte_art_model.h5")
    else:
        print("\n❌ Kaggle model training failed.")
