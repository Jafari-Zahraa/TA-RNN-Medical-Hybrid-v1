# MIMICTrainEvaluate.py (اصلاح شده)
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, fbeta_score
from MIMICTaRnnModel import TA_RNN_Medical_Hybrid
from pandas import read_csv

from GlobalVariablesMIMIC import (
    label_train, label_test,
    lon_data_train, lon_data_test,
    time_train, time_test,
    dem_data_train, dem_data_test,
    future_time_s, time_steps, embedding_matrix
)

X_train = lon_data_train[0]
y_train = label_train[0]
X_test = lon_data_test[0]
y_test = label_test[0]
demo_train = np.array(dem_data_train[0])
demo_test = np.array(dem_data_test[0])
t_train = time_train
t_train = np.reshape(t_train, (t_train.shape[0], t_train.shape[1] * t_train.shape[2]))
t_test = time_test
t_test = np.reshape(t_test, (t_test.shape[0], t_test.shape[1] * t_test.shape[2]))
# HP
file_name = 'mimic_hp_df.csv'
TA_RNN_hp_df = read_csv(file_name, header=0)
hp_list = list(TA_RNN_hp_df.iloc[0, :])

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, fbeta_score
from MIMICTaRnnModel import TA_RNN_Medical_Hybrid


def do_TA_RNN(
        X_train, t_train, y_train,
        X_test, t_test, y_test,
        demo_train, demo_test,
        iteration, hp_list,
        embedding_matrix
):
    batch_size = int(hp_list[0])
    epochs = int(hp_list[1])
    drout = hp_list[2]
    L2 = hp_list[3]
    hidden_s = int(hp_list[5])
    cell = hp_list[4].strip()
    num_icd_per_visit = X_train.shape[-1]

    # -----------------------
    # Create model
    # -----------------------
    model = TA_RNN_Medical_Hybrid(
        embedding_matrix=embedding_matrix,
        cell=cell,
        drout=drout,
        L2=L2,
        hidden_s=hidden_s,
        num_icd_per_visit=num_icd_per_visit
    )

    # -----------------------
    # Callbacks
    # -----------------------
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_dir = "saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"TA_RNN_best_iter_{iteration}.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # -----------------------
    # Train
    # ⚠️ حتماً validation_data بده اگر patient-level val داری
    # -----------------------
    model.fit(
        [X_train, t_train, demo_train],
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop, checkpoint],
        shuffle=True
    )

    # -----------------------
    # Save final model
    # -----------------------
    final_model_path = os.path.join(checkpoint_dir, f"TA_RNN_iter_{iteration}.h5")
    model.save(final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

    # -----------------------
    # Evaluate
    # -----------------------
    test_pred = model.predict([X_test, t_test, demo_test], verbose=0)
    test_pred = test_pred.reshape(-1)
    y_test_flat = y_test.reshape(-1).astype(int)

    test_pred_bin = (test_pred > 0.5).astype(int)

    acc = accuracy_score(y_test_flat, test_pred_bin)
    auc = roc_auc_score(y_test_flat, test_pred)
    f2 = fbeta_score(y_test_flat, test_pred_bin, beta=2)

    metrics_df = pd.DataFrame({
        f"Iteration {iteration}": [
            round(acc, 3),
            round(auc, 3),
            round(f2, 3)
        ]
    })

    return metrics_df, final_model_path



# =====================================================
# Main execution
# =====================================================
if __name__ == "__main__":

    iteration = 1
    hp_list = [32, 100, 0.3, 1e-4, "GRU", 128]  # Example hyperparameters: batch_size, epochs, dropout, L2, cell, hidden_s

    metrics_df, model_path = do_TA_RNN(
        X_train=X_train,
        t_train=t_train,
        y_train=y_train,
        X_test=X_test,
        t_test=t_test,
        y_test=y_test,
        demo_train=demo_train,
        demo_test=demo_test,
        iteration=iteration,
        hp_list=hp_list,
        embedding_matrix=embedding_matrix
    )

    print("✅ Training Done")
    print(metrics_df)
