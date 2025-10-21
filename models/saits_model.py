import os
import numpy as np
from pypots.imputation import SAITS

class SAITSModel:
    def __init__(self,
                 n_steps,
                 n_features,
                 device='cuda',
                 save_path='saits_model.pth',
                 epochs=500,
                 batch_size=20):
        self.save_path = save_path
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

    def fit(self, incomplete_data):
        """train model"""
        inputs = {"X": incomplete_data}
        self.model.fit(inputs)

    def impute(self, incomplete_data):
        """Perform the completion operation"""
        inputs = {"X": incomplete_data}
        imputed = self.model.impute(inputs)
        return imputed

    def exists(self):
        """判断模型是否存在"""
        return os.path.exists(self.save_path)
