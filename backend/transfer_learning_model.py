#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class TransferLearningLatteArtClassifier:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.class_names = ['heart', 'tulip', 'swan', 'rosetta']
        self.model = None
        # Try to load the trained model first, then fallback to simple model
        model_paths = [
            'kaggle_latte_art_model.h5',
            os.path.join(os.path.dirname(__file__), 'kaggle_latte_art_model.h5'),
            'transfer_latte_art_model.h5',
            os.path.join(os.path.dirname(__file__), 'transfer_latte_art_model.h5')
        ]
        
        model_loaded = False
        for model_path in model_paths:
            print(f"🔍 Checking model path: {model_path}")
            print(f"🔍 Path exists: {os.path.exists(model_path)}")
            if os.path.exists(model_path):
                print(f"🔍 File size: {os.path.getsize(model_path)} bytes")
            try:
                if os.path.exists(model_path):
                    self.load_model(model_path)
                    print(f"✅ Loaded trained model from {model_path}")
                    model_loaded = True
                    break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                import traceback
                print(f"❌ Full error: {traceback.format_exc()}")
                continue
        
        if not model_loaded:
            print("⚠️  No trained model found, will use simple model")
            print("🔍 This means the model file is not in the Docker container!")
    
    def create_transfer_model(self):
        """Create a model using transfer learning with MobileNetV2"""
        print("🏗️  Creating transfer learning model...")
        
        # Load pre-trained MobileNetV2 (without top layers)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Handle both file paths and image arrays
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return None
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, self.img_size)
        
        # Normalize pixel values (MobileNetV2 expects values between -1 and 1)
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """Predict the class of an image"""
        if self.model is None:
            # Create a simple fallback model for deployment
            self.create_simple_model()
        
        print(f"🔍 Model type: {type(self.model)}")
        print(f"🔍 Model classes: {self.class_names}")
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            print("❌ Image preprocessing failed")
            return 'other', 0.5
        
        print(f"🔍 Processed image shape: {processed_image.shape}")
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        print(f"🔍 Raw predictions: {predictions[0]}")
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = self.class_names[predicted_class_idx]
        
        print(f"🔍 Predicted class index: {predicted_class_idx}")
        print(f"🔍 Predicted class: {predicted_class}")
        print(f"🔍 Confidence: {confidence:.4f}")
        
        return predicted_class, confidence
    
    def create_simple_model(self):
        """Create a simple CNN model for deployment"""
        print("🏗️  Creating simple CNN model...")
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights (this will give random predictions)
        # In a real deployment, you'd want to train this model
        self.model = model
        print("✅ Simple model created (untrained - will give random predictions)")
    
    def train(self, data_dir, epochs=50, batch_size=16, validation_split=0.2):
        """Train the transfer learning model"""
        print("🔄 Preparing training data for transfer learning...")
        
        # Create model if it doesn't exist
        if self.model is None:
            self.create_transfer_model()
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./127.5 - 1.0,  # MobileNetV2 normalization
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"📊 Training on {len(train_generator)} batches")
        print(f"📊 Validating on {len(validation_generator)} batches")
        
        # Train the model
        print("🚀 Starting transfer learning training...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=8, factor=0.5),
                ModelCheckpoint('best_transfer_model.h5', save_best_only=True)
            ]
        )
        
        print("✅ Transfer learning training completed!")
        return history
    
    def fine_tune(self, data_dir, epochs=20, batch_size=16, validation_split=0.2):
        """Fine-tune the model by unfreezing some layers"""
        print("🔧 Fine-tuning the model...")
        
        if self.model is None:
            print("❌ No model to fine-tune!")
            return None
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 30
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🔧 Fine-tuning from layer {fine_tune_at}")
        
        # Data augmentation for fine-tuning
        train_datagen = ImageDataGenerator(
            rescale=1./127.5 - 1.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Fine-tune
        print("🚀 Starting fine-tuning...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        print("✅ Fine-tuning completed!")
        return history
    
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            if os.path.exists(model_path):
                print(f"🔍 Attempting to load model from {model_path}")
                print(f"🔍 TensorFlow version: {tf.__version__}")
                
                # Try loading with different options
                try:
                    # First try: standard load
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"✅ Model loaded successfully with standard method")
                except Exception as e1:
                    print(f"❌ Standard load failed: {e1}")
                    try:
                        # Second try: load with compile=False
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        print(f"✅ Model loaded successfully with compile=False")
                    except Exception as e2:
                        print(f"❌ Load with compile=False failed: {e2}")
                        try:
                            # Third try: load with custom objects
                            self.model = tf.keras.models.load_model(model_path, custom_objects={'MobileNetV2': tf.keras.applications.MobileNetV2})
                            print(f"✅ Model loaded successfully with custom objects")
                        except Exception as e3:
                            print(f"❌ All load methods failed: {e3}")
                            raise e3
                
                print(f"📥 Model loaded from {model_path}")
                print(f"🔍 Loaded model type: {type(self.model)}")
            else:
                print(f"📝 No existing model found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            self.model = None

def main():
    """Train a transfer learning model"""
    print("🚀 Training Transfer Learning Model")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists("dataset_augmented"):
        print("❌ Augmented dataset not found!")
        print("💡 Run: python3 augment_dataset.py first")
        return
    
    # Initialize classifier
    classifier = TransferLearningLatteArtClassifier()
    
    # Train the model
    print("🎓 Training transfer learning model...")
    history = classifier.train(
        data_dir="dataset_augmented",
        epochs=50,
        batch_size=16
    )
    
    # Fine-tune the model
    print("\n🔧 Fine-tuning the model...")
    fine_tune_history = classifier.fine_tune(
        data_dir="dataset_augmented",
        epochs=20,
        batch_size=16
    )
    
    # Save the final model
    classifier.save_model("transfer_latte_art_model.h5")
    
    print("\n🎉 Transfer learning training completed!")
    print("💡 This model should perform much better than the previous one!")

if __name__ == "__main__":
    main()
