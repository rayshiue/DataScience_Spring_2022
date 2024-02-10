- Folder structure
    ```
    .
    ├── dataset               #Put the data here     
    │   ├── train_sub-graph_tensor.pt
    │   ├── test_sub-graph_tensor_noLabel.pt
    │   ├── train_mask.npy
    │   └── test_mask.npy
    ├── train.py
    ├── test.py
    ├── requirements.txt
    └── Readme.md
    ```

## Download Required Pakages (Using Python 3.9)
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
1.執行前需要先把 dataset 資料夾放進當前目錄，dataset中有包含train跟test的各自的graph_tensor及mask檔。
2.接著bash 0810892.sh，會輸出預測的結果0810892.csv。
3.若要重現訓練過程，則直接執行train.py，執行完畢後會用新訓練的模型覆蓋掉當前的best.pth。