- Folder structure
    ```
    .
    ├── hw5_dataset               #Put the data here     
    │   ├── train.json         
    │   └── test.json  
    ├── utils.py
    ├── train.py
    ├── test.py
    ├── requirements.txt
    └── Readme.md
    ```

## Download Required Pakages
```sh
pip install -r requirement.txt
```

## Make Predictions
```sh
bash 0810892.sh
```

## Train
```sh
python train.py
```

---執行說明---
1.執行前需要先把 hw5_dataset 資料夾放進當前目錄，hw5_dataset中有train.json以及test.json兩個檔案。
2.接著bash 0810892.sh，會輸出預測的結果0810892.json。(會先將訓練好的模型從hugging face載下來)
3.若要重現訓練過程，則直接執行train.py，總共會執行20個epochs，每個epoch會做一次evaluate。