- Folder structure
    ```
    .
    ├── data               #Put data here     
    │   ├── train         
    │   └── test  
    ├── train.py
    ├── test.py
    ├── vgg.py
    ├── requirements.txt
    ├── best_model.pth
    └── Readme.md
    ```

## Make Prediction
```sh
bash 0810892.sh
```

## Train (Optional)
```sh
python train.py
```

The prediction file is `output.csv`.


---執行說明---
1.執行前需要先把 data 資料夾放進當前目錄，data中有train以及test兩個資料夾。
2.接著bash 0810892.sh，會輸出預測的結果output.csv。
3.若要重現訓練過程，則直接執行train.py，總共會執行100個epochs。