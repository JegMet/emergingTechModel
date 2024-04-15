from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np

def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Assuming your model expects pixel values to be scaled
    return img_array


# Load the model
model = load_model("D:\EmergingTechFinalProject\emergingTechFinal\\newmodel.keras")

# Preprocess the image
img_path = 'D:\EmergingTechFinalProject\emergingTechFinal\\PetImages\Cat\\10.jpg'
processed_image = preprocess_image(img_path, 180, 180)

# Make predictions
predictions = model.predict(processed_image)
print(predictions)
predicted_class = np.argmax(predictions, axis=1)



# Assuming you know the class names
class_names = ['Cat', 'Dog']
predicted_class_name = class_names[predicted_class[0]]
print(f'The predicted class name is: {predicted_class_name}')
