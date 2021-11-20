from pandas import np
from tensorflow.keras.applications.resnet50 import preprocess_input

#restful预测

#pred_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
from keras.preprocessing.image import ImageDataGenerator

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input
)

# re-size all the images to this
IMAGE_SIZE = [100, 100]
valid_path = 'E:\\tmp\\archive\\fruits-360_dataset\\fruits-360\\Test'

valid_gen = val_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)

#对水果类别名称建立数组，数组数量为类别数量
labels = [None] * len(valid_gen.class_indices)
for k, v in valid_gen.class_indices.items():
  labels[v] = k


pred_path = 'E:\\tmp\\archive\\fruits-360_dataset\\fruits-360\\pred'

pred_datagen = ImageDataGenerator()
pred_generator = pred_datagen.flow_from_directory(pred_path, target_size=(100, 100),batch_size=1,class_mode='categorical', shuffle=False,)
pred_generator.reset()

for i in range(len(pred_generator)):
    value = pred_generator.next()
    #print(value)
    import json
    data = json.dumps({"signature_name": "serving_default", "instances": value[0].tolist()})

#通过restful api 调用预测
    import requests
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/tfServing_Fruit:predict', data=data, headers=headers)

    predictions = json.loads(json_response.text)['predictions']
    for pred in predictions:
        print(np.argmax(pred))
        print(labels[np.argmax(pred)])