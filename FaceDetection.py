import os
import glob
import numpy as np
import keras
from keras.callbacks import EarlyStopping

#%%
dict_labels = {"norm":0, "happy":1,"bored":2,"confuse":3,"focus":4,"frustration":5,"suprise":6}

size = (48,48) #size一致統一為48*48

#%% 
labels=['norm','happy','bored','confuse','focus','frustration','suprise']
base_path = r'C:\Users\qwasd\OneDrive\桌面\123'
layers_of_folders=0
folder_list=[]
if base_path :
    folder_layers=[]
    files = os.scandir(base_path)
    #第一層路徑
    first_folder = []
    first_folder_kind = []
    for entry in files:
        if entry.is_dir():
            first_folder.append(entry.path)
            first_folder_kind.append(entry.name)
    folder_layers.append(first_folder_kind)
    folder_list.append(first_folder)
    #第二層路徑
    second_folder = []
    if first_folder:
        second_folder = []
        second_folder_kind = []
        layers_of_folders+=1
        for fldr in first_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    second_folder.append(entry.path)
                    second_folder_kind.append(entry.name)
        second_folder_kind= second_folder_kind[0:int(len(second_folder_kind)/len(first_folder_kind))]
        folder_layers.append(second_folder_kind)
        folder_list.append(second_folder)
    #第三層路徑
    third_folder = []
    if second_folder:
        third_folder = []
        third_folder_kind = []
        layers_of_folders+=1
        for fldr in second_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    third_folder.append(entry.path)
                    third_folder_kind.append(entry.name)
        third_folder_kind= third_folder_kind[0:int(len(third_folder_kind)/(len(second_folder_kind)*len(first_folder_kind)))]
        folder_list.append(third_folder)
    #第四層路徑
    forth_folder = []
    if third_folder:
        forth_folder = []
        forth_folder_kind = []
        layers_of_folders+=1
        for fldr in third_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    forth_folder.append(entry.path)
                    forth_folder_kind.append(entry.name)
        forth_folder_kind= forth_folder_kind[0:int(len(forth_folder_kind)/(len(third_folder_kind)*len(second_folder_kind)*len(first_folder_kind)))]
        folder_list.append(forth_folder)
     #第五層路徑
    if forth_folder:
        fifth_folder = []
        fifth_folder_kind = []
        layers_of_folders+=1
        for fldr in third_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    fifth_folder.append(entry.path)
                    fifth_folder_kind.append(entry.name)
        fifth_folder_kind= fifth_folder_kind[0:int(len(fifth_folder_kind)/(len(forth_folder_kind)*len(third_folder_kind)*len(second_folder_kind)*len(first_folder_kind)))]
#%%
#datanumber=nbofdata
blob=[]
blob_nparray=[]
image_data=[]
conc = 0
labels_dict={}
for entry1 in folder_list[layers_of_folders - 1]:
    blob = []
    cellname = os.path.basename(os.path.dirname(entry1)) 
    print(cellname)
    concnames = os.path.basename(entry1)
    print(concnames)
    if concnames in labels:
        labels_dict[conc] = concnames
        fnamelist = glob.glob(os.path.join(entry1, '*.jpg'))
        for filename in fnamelist[0:]:
            im = Image.open(filename)
            if im is not None:
                if im.mode=='RGB':
                    im=im.resize(size,Image.BILINEAR)
                    imarray = np.array(im)
                    blob.append(imarray)
        ind = np.reshape(np.arange(1, len(blob) + 1), (-1, 1))
        blob_nparray = np.reshape(np.asarray(blob), (len(blob), blob[1].size))
        blob_nparray = np.hstack((blob_nparray, ind, conc * np.ones((len(blob), 1))))
        image_data.append(np.asarray(blob_nparray, dtype=np.float32))
        print(concnames+'  finished!')
        conc += 1
#%%
for j in range(len(labels)):
    trytry=image_data[j][:]
#資料預處理
    LengthT = trytry.shape[0]
    trytry_index = trytry[...,-2:-1]
    trytry_label = trytry[...,-1:] 
    trytry = trytry[...,:-2]
    
#正規化
    trytry -= np.reshape(np.mean(trytry, axis=1), (-1,1))
    trytry = np.reshape(trytry, (trytry.shape[0],48,48,3))
    trytry = trytry.reshape(-1,48,48,3)
    np.random.shuffle(trytry)
    trytry_train_upto = round(trytry.shape[0] * 8 / 10)
    trytry_test_upto = trytry.shape[0]
    if j is 0:
        train_data = trytry[:trytry_train_upto]
        test_data = trytry[trytry_train_upto:trytry_test_upto]
        train_label = trytry_label[:trytry_train_upto]
        test_label = trytry_label[trytry_train_upto:trytry_test_upto]
        
    else:
        train_data = np.concatenate((train_data, 
                                     trytry[:trytry_train_upto]), axis=0)
        
        test_data = np.concatenate((test_data, 
                                    trytry[trytry_train_upto:trytry_test_upto]), axis=0)
        
        train_label = np.concatenate((train_label, 
                                     trytry_label[:trytry_train_upto]), axis=0)
        
        
        test_label = np.concatenate((test_label, 
                                    trytry_label[trytry_train_upto:trytry_test_upto]), axis=0)
        
test_label = keras.utils.to_categorical(test_label, num_classes=len(labels))
train_label = keras.utils.to_categorical(train_label, num_classes=len(labels))
print(train_data.shape)
print(test_data.shape)
#print(train_label[900:1200])

train_data=train_data.astype('float32')/255.0
test_data=test_data.astype('float32')/255.0
#%% 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48,3),padding='same',name='block1_conv2_1'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu',name='final_output_1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu',name='final_output_2'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid',name='class_output'))

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
EStop = EarlyStopping(monitor='val_acc', min_delta=0, 
                      patience=10, verbose=1, mode='auto')
print(model.summary())
#%% 訓練模型  80%訓練 20%驗證
his = model.fit(train_data, train_label, batch_size=128, epochs=100,shuffle=True, validation_split=0.2,callbacks=[EStop])
#%%
model.save('catdog_model.h5') 

#混淆矩陣
from sklearn.metrics import confusion_matrix
predict_classes = model.predict_classes(test_data[1:,])
true_classes = np.argmax(test_label[1:],1)
print(predict_classes.shape)
print(true_classes.shape)
print(confusion_matrix(true_classes, predict_classes))
