from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Flatten , GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import os
#import splitfolders

input_folder = "C:/Users/saady/OneDrive/Documents/chatbot/imges"
output_dir = "C:/Users/saady/OneDrive/Documents/chatbot/processing"

#splitfolders.ratio(input_folder, output=output_dir, seed=1337, ratio=(.7, .15, .15))

img_height,img_width=(224,224)

def res_imgs(input_folder, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                img = img.resize(size, Image.LANCZOS)
                img=np.array(img)
                img = np.expand_dims(x, axis=1)    
                img.save(os.path.join(output_dir, filename))
                print(f"Resized {filename}")
            except Exception as e:
                print(f"Could not resize {filename}: {e}")

res_imgs(input_folder,output_dir,(img_height,img_width))

train_data ="C:/Users/saady/OneDrive/Documents/chatbot/processing/train"
test_data="C:/Users/saady/OneDrive/Documents/chatbot/processing/test"
vali_data="C:/Users/saady/OneDrive/Documents/chatbot/processing/val"



datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=20,         
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True
)


img_height,img_width=(224,224)

# train_generator_aug = datagen.flow_from_directory(
#     train_data,                
#     target_size=(img_height, img_width),    
#     class_mode="categorical", 
#     batch_size=32,             
#     shuffle=True
# )

# test_generator_aug = datagen.flow_from_directory(
#     test_data,                
#     target_size=(img_height, img_width),    
#     class_mode="categorical", 
#     batch_size=32,             
#     shuffle=True
# )

# val_generator_aug = datagen.flow_from_directory(
#     vali_data,                
#     target_size=(img_height, img_width),    
#     class_mode="categorical", 
#     batch_size=32,             
#     shuffle=True
# )

x, y = next(train_generator_aug)
print(x.shape)
len(train_generator_aug)


base_model = ResNet50(include_top=False, weights="imagenet")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

num_classes = 24
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()


model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)


history = model.fit(
    train_generator_aug,    
    validation_data=val_generator_aug,  
    epochs=15,                 
    steps_per_epoch=len(train_generator_aug),  
    validation_steps=len(val_generator_aug)    
)

model_json = model.to_json()
with open("model_arch.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("my_model.weights.h5")


