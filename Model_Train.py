import tensorflow as tf
import numpy as np
from PIL import Image
import os

class TomatoDiseaseClassifier:
    def __init__(self, model_path="tomato_model_v2.keras"):
        """
        Initialize the Tomato Disease Classifier.
        
        Args:
            model_path: Path to the trained model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Early_blight', 'Late_blight', 'Leaf_Mold', 'healthy']
        self.img_size = (224, 224)
        print("Model loaded successfully!")
    
    def predict_image(self, image_path, show_image=False):
        """
        Predict the disease from a single image.
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image (default: False)
            
        Returns:
            dict: Prediction results including class and confidence
        """
        try:
            # Load and preprocess image
            img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
            
            if show_image:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Input Image: {os.path.basename(image_path)}")
                plt.show()
            
            # Convert to array and prepare for prediction
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            
            # Get results
            predicted_class = self.class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            # Get all class probabilities
            probabilities = {self.class_names[i]: 100 * score[i].numpy() 
                           for i in range(len(self.class_names))}
            
            result = {
                'image': os.path.basename(image_path),
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'is_healthy': predicted_class == 'healthy',
                'advice': self.get_advice(predicted_class)
            }
            
            self.print_results(result)
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def get_advice(self, disease_class):
        """Get agricultural advice based on disease prediction."""
        advice_dict = {
            'Early_blight': """ADVICE FOR EARLY BLIGHT:
            • Remove infected leaves immediately
            • Apply fungicides containing chlorothalonil or copper
            • Improve air circulation around plants
            • Avoid overhead watering
            • Rotate crops next season""",
            
            'Late_blight': """ADVICE FOR LATE BLIGHT:
            • URGENT: This is highly contagious
            • Remove and destroy all infected plants
            • Apply fungicides (mancozeb, metalaxyl)
            • Do not compost infected plants
            • Keep foliage as dry as possible""",
            
            'Leaf_Mold': """ADVICE FOR LEAF MOLD:
            • Reduce humidity in greenhouse
            • Increase ventilation
            • Remove infected leaves
            • Apply sulfur-based fungicides
            • Space plants properly for air flow""",
            
            'healthy': """PLANT IS HEALTHY:
            • Continue regular monitoring
            • Maintain proper watering schedule
            • Ensure adequate sunlight (6-8 hours daily)
            • Apply balanced fertilizer monthly
            • Watch for early signs of disease"""
        }
        return advice_dict.get(disease_class, "No specific advice available.")
    
    def print_results(self, result):
        """Print formatted prediction results."""
        print("\n" + "="*50)
        print("TOMATO LEAF DISEASE CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nProbabilities:")
        for disease, prob in result['probabilities'].items():
            print(f"  • {disease}: {prob:.2f}%")
        print("\n" + result['advice'])
        print("="*50 + "\n")
    
    def batch_predict(self, image_dir):
        """
        Predict diseases for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            list: Predictions for all images
        """
        if not os.path.exists(image_dir):
            print(f"Directory {image_dir} not found.")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = [f for f in os.listdir(image_dir) 
                 if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not images:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"Found {len(images)} images in {image_dir}")
        results = []
        
        for img_file in images:
            img_path = os.path.join(image_dir, img_file)
            result = self.predict_image(img_path, show_image=False)
            if result:
                results.append(result)
        
        return results

def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tomato Leaf Disease Classifier')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Directory containing multiple images')
    parser.add_argument('--model', type=str, default='tomato_model_v2.keras', 
                       help='Path to model file (default: tomato_model_v2.keras)')
    
    args = parser.parse_args()
    
    try:
        # Initialize classifier
        classifier = TomatoDiseaseClassifier(args.model)
        
        if args.image:
            # Predict single image
            classifier.predict_image(args.image, show_image=True)
        
        elif args.dir:
            # Predict all images in directory
            results = classifier.batch_predict(args.dir)
            
            # Print summary
            if results:
                print("\n" + "="*50)
                print("BATCH PREDICTION SUMMARY")
                print("="*50)
                for result in results:
                    status = "✓" if result['is_healthy'] else "✗"
                    print(f"{status} {result['image']}: {result['prediction']} ({result['confidence']:.1f}%)")
        
        else:
            # Interactive mode
            print("Tomato Disease Classifier - Interactive Mode")
            print("Enter 'q' to quit")
            
            while True:
                image_path = input("\nEnter image path: ").strip()
                if image_path.lower() == 'q':
                    break
                
                if os.path.exists(image_path):
                    classifier.predict_image(image_path, show_image=True)
                else:
                    print("File not found. Please enter a valid path.")

    except FileNotFoundError as e:
        print(e)
        print("\nTo train a model, run: python train_model.py")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()