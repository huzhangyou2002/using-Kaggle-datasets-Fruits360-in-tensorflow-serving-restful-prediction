import os

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from glob import glob

import warnings
warnings.filterwarnings("ignore")

def save_model(model,dir_str):
    #MODEL_DIR = "E:\\tmp\\tfserving\\"
    version = 1
    export_path = os.path.join(dir_str, str(version))
    #print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model Finished')

def construct_model(folders):
    #使用imagenet的 RestNet50模型
    res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    #定义其中层，都不训练
    for layer in res.layers:
        layer.trainable = False

    #对ResNet50模型中的output进行 Flatten操作
    x = Flatten()(res.output)

    #因为是一个分类问题
    #预测结果为对上诉x 进行一个softmax激活函数的Dense操作，对应的类别数量为文件夹数量
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    #创建模型，输入为res,输出为prediction
    model = Model(inputs=res.input, outputs=prediction)

    return model


# re-size all the images to this
IMAGE_SIZE = [100, 100]

# training config:
epochs = 16
batch_size = 128

train_path = 'E:\\tmp\\archive\\fruits-360_dataset\\fruits-360\\Training'
valid_path = 'E:\\tmp\\archive\\fruits-360_dataset\\fruits-360\\Test'


# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')

#构造模型
model = construct_model(folders)
#model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# create an instance of ImageDataGenerator
train_gen = ImageDataGenerator(
  #随机对图片旋转20度
  rotation_range=20,
  #如果 <1，则是除以总宽度的值
  width_shift_range=0.1,
  #如果 <1，则是除以总宽度的值
  height_shift_range=0.1,
  #所谓shear_range就是错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移，且平移的大小和该点到x轴(或y轴)的垂直距离成正比
  shear_range=0.1,
  #随机缩放范围。而参数大于0小于1时，执行的是放大操作
  zoom_range=0.2,
  #随机水平翻转。
  horizontal_flip=True,
  #随机垂直翻转
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input
)

# create generators
train_generator = train_gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
  class_mode='sparse',
)
valid_generator = val_gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=batch_size,
  class_mode='sparse',
)


# fit the model
r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)

MODEL_DIR = "E:\\tmp\\tfServing_Fruit\\"
save_model(model,MODEL_DIR)
