import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def print_folders(path):
    cnt = 0
    for name in os.listdir(path):
        print(name)
        cnt += 1
    print("Total number of folders:", cnt)

# Define image size and batch size
IMAGE_SIZE = (45, 45)  # Your image size is 45x45
BATCH_SIZE = 32  # Define a reasonable batch size

# Define the path to your dataset
dataset_dir = "data/extracted_images"

# Create an ImageDataGenerator for training data with rescaling and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # 80% for training, 20% for validation
    rotation_range=10,  # Small rotation augmentation
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1  # Vertical shift
)

# Load the training set
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',  # If the images are in grayscale
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Multiclass classification
    subset='training'  # Use 80% of data for training
)

# Load the validation set
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Use 20% of data for validation
)

# Check the class indices (this maps folder names to numeric labels)
print(train_generator.class_indices)
