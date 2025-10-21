# Missing-traffic-data-completion-based-on-PyPOTS-and-visualization-using-Dash-Plotly



## Introduction

This project uses the **SAITS** model from [PyPOTS](https://github.com/WenjieDu/PyPOTS) to perform **time series imputation**, filling in missing values within temporal data. The dataset used is a small private collection that is not publicly available. In addition, the project employs **Dash** and **Plotly** to create an interactive visualization interface that compares the complete time series with the randomly cropped (incomplete) version.

## Private traffic data

### original data

The traffic data has a shape of (228, 12672), where 228 represents the number of time steps and each time step has 12,672 features. The figure below shows the variation of a single feature across different time steps.

<img src="D:\主业\github\work3.9的代码环境\traffic1021\pic\originaldata.png" alt="1" style="zoom:30%;" />

### incomplete data after random cropping

the complete time series data was randomly cropped to create traffic time series data for training the SAITS model. The core code is shown below:

```python
ata = arr[np.newaxis, :, :]
rng = np.random.default_rng(42)
mask = rng.random(data.shape) < MISSING_RATE
incomplete = data.copy()
incomplete[mask] = np.nan
```

<img src="D:\主业\github\work3.9的代码环境\traffic1021\pic\incompletedata2.png" alt="1" style="zoom:30%;" />

<img src="D:\主业\github\work3.9的代码环境\traffic1021\pic\incompletedata.png" alt="1" style="zoom:30%;" />

## SAITS of PyPOTS

### download PyPOTS

```cmd
conda install pypots
```

or

```cmd
pip install pypots
```



### model 's structure ( see models/saits_model.py)

The SAITS model based on PyPOTS is defined in the file models/saits_model.py.

```python
class SAITSModel:
    def __init__(self, n_steps, n_features, device, save_path, epochs, batch_size):
        self.model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=3,
            d_model=64,
            n_heads=2,
            d_k=32,
            d_v=32,
            d_ffn=128,
            dropout=0.1,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            saving_path=save_path
        )

    def fit(self, incomplete):
        inputs = {"X": incomplete}
        self.model.fit(inputs)
        # 保存训练曲线
        self.training_history = self.model.training_history_

    def impute(self, incomplete):
        inputs = {"X": incomplete}
        return self.model.impute(inputs)
```

### trainmodel.py

In the "trainmodel.py" file, you can load the dataset to train the model.

## Interactive interface with dash/plotly to show results of SAITS

In this visualization.py, simple Dash and Plotly were used to visualize the traffic time series imputation results of the SAITS model. The result as show in picture:

<img src="D:\主业\github\work3.9的代码环境\traffic1021\pic\resultmodel.png" alt="1" style="zoom:30%;" />





