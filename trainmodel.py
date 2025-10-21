import os
import numpy as np
import matplotlib.pyplot as plt
from models.saits_model import SAITSModel


# parameters for training
DEVICE = 'cuda'
MODEL_SAVE_PATH = "models/saits_model.pth" # where to save the best model
IMPUTED_SAVE_PATH = "imputed.npy" # where to save the prediction results of models
EPOCHS = 500
BATCH_SIZE = 20
MISSING_RATE = 0.1


'''
step 1: data processing and generate incomplete training data

Process the data and randomly cut off certain segments of the time series 
to form incomplete time series data.
'''

arr = np.load("datas/pems.npy")
print("original data shape:", arr.shape)

data = arr[np.newaxis, :, :]

rng = np.random.default_rng(42)
mask = rng.random(data.shape) < MISSING_RATE
incomplete = data.copy()
incomplete[mask] = np.nan
print("incomplete data shape:", incomplete.shape)


'''
Step 2: training....
'''

# if there already is a training results, just use it and do not need to retrain.
if os.path.exists(IMPUTED_SAVE_PATH):
    print("It has been detected that there are already time series completion results available. Proceed to load them directly...")
    imputed = np.load(IMPUTED_SAVE_PATH)
else:
    print("start train SAITS model...")
    model = SAITSModel(
        n_steps=data.shape[1],
        n_features=data.shape[2],
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.fit(incomplete)
    imputed = model.impute(incomplete)

    np.save(IMPUTED_SAVE_PATH, imputed)
    print(f"The time series completion results are saved to {IMPUTED_SAVE_PATH}")


'''
Save Matplotlib visualization as static picture.
'''

# Matplotlib visualization and it is saved as a picture
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(incomplete[0, :, 0], "o-", label="With Missing", markersize=3)
axes[0].set_title("Missing Data")
axes[0].legend()

axes[1].plot(imputed[0, :, 0], "--", label="Imputed", linewidth=1)
axes[1].set_title("Imputed Data by SAITS")
axes[1].legend()

axes[2].plot(data[0, :, 0], label="Original", linewidth=1)
axes[2].set_title("Original Complete Data")
axes[2].legend()

plt.tight_layout()
plt.savefig("imputation_overview.png", dpi=600)
plt.close()
print('The Matplotlib visualization has been saved as "imputation_overview.png"')


# ===============================
# Save Matplotlib visualization as static picture.
# ===============================

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(incomplete[0, :, 0], "o-", label="With Missing", markersize=3)
axes[0].set_title("Missing Data")
axes[0].legend()

axes[1].plot(imputed[0, :, 0], "--", label="Imputed", linewidth=1)
axes[1].set_title("Imputed Data by SAITS")
axes[1].legend()

axes[2].plot(data[0, :, 0], label="Original", linewidth=1)
axes[2].set_title("Original Complete Data")
axes[2].legend()

plt.tight_layout()
plt.savefig("imputation_overview.png", dpi=600)
plt.close()
print('The Matplotlib visualization has been saved as "imputation_overview.png"')


