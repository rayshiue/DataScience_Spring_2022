- Folder structure
    ```
    .
    ├── train.py
    ├── eval.py
    ├── finetune.py
    ├── special_resnet.py
    ├── transforms.py
    ├── resnet-50.pth (需要先把這個模型放進來!!!!!!!!)
    ├── pruned_model.pth (原本就有，但train完之後會被新的模型覆蓋掉)
    ├── Readme.md
    ├── data (執行eval或train之後會出現，存放訓練與測試資料)
    └── torch_pruning
    ```

## Make Prediction
```sh
python eval.py
```

## Train (Optional)
```sh
python train.py
```

The prediction file is `example_pred.csv`.


---執行說明---
1.執行前需要先把resnet-50.pth放進當前目錄。
2.接著執行eval.py，將會對pruned_model.pth進行test，並顯示參數量與準確度，再輸出預測的csv檔。
3.若要重現訓練過程，則需要執行train.py，將會依次進行KD、Pruning以及predict。
4.由於完整training的epochs數目過大，在3080上大約需要10小時，因此提供簡化版的train.py(約需2小時半)，但會獲得比較差的結果(約94%)，提供助教進行測試。
