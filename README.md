# AI_FaceDetection #
## 資料收集 (DataCollection) ##

 bores | confuse | focus | frustration | happy | norm | surprise  
-------|---------|-------|-------------|-------|------|----------- 
  1152 |   935   |  1057 |     710     |  826  |  754 |   949
  
 > 總共 6383 張照片，皆統一為 48*48 大小
  
## 資料分割_測試訓練集 (DataSplit_TrainTest) ##

 > 5107 張train , 1276張test
  
## 分類 (Classification) ##

 > 將照片利用 one-hot-encoding 方法，分類為七種情緒
 
## 模型訓練_CNN (TrainModel_CNN) ##

 > CNN 參數，最後 class_output 為七種情緒
 > 模型訓練，準確率最高為 96.35% 
 
## 驗證 (Validation) ##

 > 1275張做最後模型驗證
 
 
 ## 實驗說明 () ##

 (1) 發現困惑和驚訝不明顯，因為驚訝照片在臉部表情上多數沒有嘴巴張開的特徵，導致在辨試困惑時，會有差錯。
 
 (2)	挫折被誤判為困惑、投入、驚訝比例皆較高，可能在這四種情緒中，有人工標記圖片上的疏忽。
 
 (3)	在驚訝情緒中皆為0的原因，可能臉部情緒沒有太明顯之變化。
