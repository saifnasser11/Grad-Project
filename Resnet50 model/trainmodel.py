from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import scipy

#checkpoint_path = 'model_epoch_{epoch:02d}.h5'
#checkpoint = ModelCheckpoint(
#    filepath=checkpoint_path,
#    save_weights_only=True,
#    period=1
#)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    r'D:\MSA\Grad\new\Celeb-DF-v2\datatset\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
)

validation_generator = datagen.flow_from_directory(
    r'D:\MSA\Grad\new\Celeb-DF-v2\datatset\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
)

test_generator = datagen.flow_from_directory(
    r'D:\MSA\Grad\new\Celeb-DF-v2\datatset\test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)

class_labels = train_generator.classes
print(class_labels)
class_labels = test_generator.classes
print(class_labels)
class_labels = validation_generator.classes
print(class_labels)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

with tf.device("/gpu:0"):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for epoch in range(10):
    if epoch % 5 == 0:
        model.save('model_epoch_{epoch:02d}.h5')
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=10,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            # callbacks=[checkpoint]
        )

loss, accuracy = model.evaluate(test_generator)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")


resnet_loss, resnet_accuracy = model.evaluate(test_generator)

print("\nResNet Model:")
print(f"Test loss: {resnet_loss:.4f}")
print(f"Test accuracy: {resnet_accuracy * 100:.2f}%")

model.save('ResNet Model.h5')
