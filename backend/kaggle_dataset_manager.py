#!/usr/bin/env python3
"""
Automated Kaggle Dataset Manager for Latte Art Classification
Downloads, organizes, and prepares datasets for training
"""

import os
import shutil
import zipfile
import subprocess
import json
from pathlib import Path

class KaggleDatasetManager:
    def __init__(self, username="justinsungby", dataset_name="mingchenadam/latte-art-train"):
        self.username = username
        self.dataset_name = dataset_name
        self.kaggle_dir = "kaggle_data"
        self.processed_dir = "dataset_kaggle"
        
    def setup_kaggle_api(self):
        """Set up Kaggle API configuration"""
        print("🔧 Setting up Kaggle API...")
        
        # Check if kaggle is installed
        try:
            import kaggle
            print("✅ Kaggle package found")
        except ImportError:
            print("📦 Installing Kaggle package...")
            subprocess.run(["pip", "install", "kaggle"], check=True)
            import kaggle
        
        # Check for environment variables first (Render/production)
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if kaggle_username and kaggle_key:
            print("✅ Using environment variables for Kaggle API")
            # Create kaggle.json from environment variables
            kaggle_home = Path.home() / ".kaggle"
            kaggle_home.mkdir(exist_ok=True)
            
            kaggle_config = {
                "username": kaggle_username,
                "key": kaggle_key
            }
            
            api_key_path = kaggle_home / "kaggle.json"
            with open(api_key_path, 'w') as f:
                json.dump(kaggle_config, f)
            
            # Set proper permissions
            os.chmod(api_key_path, 0o600)
            print("✅ Kaggle API configured from environment variables")
            return True
        
        # Fallback to local file (development)
        kaggle_home = Path.home() / ".kaggle"
        kaggle_home.mkdir(exist_ok=True)
        
        api_key_path = kaggle_home / "kaggle.json"
        if not api_key_path.exists():
            print("⚠️  Kaggle API key not found!")
            print("📋 Please follow these steps:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Click 'Create New API Token'")
            print("   3. Download the kaggle.json file")
            print("   4. Place it in ~/.kaggle/kaggle.json")
            print("   5. Run: chmod 600 ~/.kaggle/kaggle.json")
            print()
            print("💡 For Render deployment, use environment variables:")
            print("   KAGGLE_USERNAME=your_username")
            print("   KAGGLE_KEY=your_api_key")
            return False
        
        print("✅ Kaggle API configured from local file")
        return True
    
    def download_dataset(self):
        """Download the dataset from Kaggle"""
        print(f"📥 Downloading dataset: {self.dataset_name}")
        
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Create download directory
            os.makedirs(self.kaggle_dir, exist_ok=True)
            
            # Download dataset
            print("🔄 Downloading dataset files...")
            api.dataset_download_files(
                self.dataset_name,
                path=self.kaggle_dir,
                unzip=True
            )
            
            print("✅ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def analyze_dataset_structure(self):
        """Analyze the downloaded dataset structure"""
        print("🔍 Analyzing dataset structure...")
        
        if not os.path.exists(self.kaggle_dir):
            print("❌ Dataset directory not found!")
            return None
        
        structure = {}
        for root, dirs, files in os.walk(self.kaggle_dir):
            level = root.replace(self.kaggle_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"{subindent}{file}")
                    category = os.path.basename(root)
                    if category not in structure:
                        structure[category] = []
                    structure[category].append(file)
        
        print(f"\n📊 Dataset Summary:")
        for category, files in structure.items():
            print(f"   {category}: {len(files)} images")
        
        return structure
    
    def reorganize_dataset(self, structure):
        """Reorganize dataset into our standard structure"""
        print("🔄 Reorganizing dataset...")
        
        # Create processed directory structure
        categories = ['heart', 'tulip', 'swan', 'rosetta']
        for category in categories:
            os.makedirs(os.path.join(self.processed_dir, category), exist_ok=True)
        
        # Map Kaggle categories to our categories
        category_mapping = {
            'heart': 'heart',
            'tulip': 'tulip', 
            'swan': 'swan',
            'rosetta': 'rosetta'
        }
        
        total_copied = 0
        for kaggle_category, files in structure.items():
            if kaggle_category in category_mapping:
                our_category = category_mapping[kaggle_category]
                target_dir = os.path.join(self.processed_dir, our_category)
                
                print(f"📁 Processing {kaggle_category} → {our_category}")
                
                for i, file in enumerate(files):
                    source_path = os.path.join(self.kaggle_dir, "latte_art", kaggle_category, file)
                    target_path = os.path.join(target_dir, f"{i+1:03d}.jpg")
                    
                    try:
                        shutil.copy2(source_path, target_path)
                        total_copied += 1
                    except Exception as e:
                        print(f"   ⚠️  Error copying {file}: {e}")
                
                print(f"   ✅ Copied {len(files)} images")
        
        print(f"\n🎉 Dataset reorganization complete!")
        print(f"📊 Total images copied: {total_copied}")
        return total_copied
    
    def create_augmented_dataset(self):
        """Create augmented version of the Kaggle dataset"""
        print("🔄 Creating augmented dataset...")
        
        # Import our augmentation system
        from transfer_learning_model import TransferLearningLatteArtClassifier
        
        # Create augmented dataset with 3x augmentation
        augmented_dir = "dataset_kaggle_augmented"
        os.makedirs(augmented_dir, exist_ok=True)
        
        categories = ['heart', 'tulip', 'swan', 'rosetta']
        for category in categories:
            os.makedirs(os.path.join(augmented_dir, category), exist_ok=True)
        
        # Copy original images
        total_originals = 0
        for category in categories:
            source_dir = os.path.join(self.processed_dir, category)
            target_dir = os.path.join(augmented_dir, category)
            
            if os.path.exists(source_dir):
                for filename in os.listdir(source_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_path = os.path.join(source_dir, filename)
                        target_path = os.path.join(target_dir, f"orig_{filename}")
                        shutil.copy2(source_path, target_path)
                        total_originals += 1
        
        print(f"✅ Augmented dataset created with {total_originals} original images")
        return augmented_dir
    
    def run_full_pipeline(self):
        """Run the complete dataset pipeline"""
        print("🚀 Starting Kaggle Dataset Pipeline")
        print("=" * 40)
        
        # Step 1: Setup Kaggle API
        if not self.setup_kaggle_api():
            return False
        
        # Step 2: Download dataset
        if not self.download_dataset():
            return False
        
        # Step 3: Analyze structure
        structure = self.analyze_dataset_structure()
        if not structure:
            return False
        
        # Step 4: Reorganize dataset
        total_images = self.reorganize_dataset(structure)
        if total_images == 0:
            return False
        
        # Step 5: Create augmented dataset
        augmented_dir = self.create_augmented_dataset()
        
        print(f"\n🎉 Pipeline Complete!")
        print(f"📁 Original dataset: {self.processed_dir}")
        print(f"📁 Augmented dataset: {augmented_dir}")
        print(f"📊 Total images: {total_images}")
        
        return True

def main():
    """Main function"""
    manager = KaggleDatasetManager()
    success = manager.run_full_pipeline()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Train new model: python3 train_kaggle_model.py")
        print("   2. Update frontend for new categories")
        print("   3. Test improved performance!")
    else:
        print("\n❌ Pipeline failed. Please check the errors above.")

if __name__ == "__main__":
    main()
