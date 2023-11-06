import os
import time
import pathlib
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score,recall_score,f1_score
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from util.model import DCNN
from util.data import load_hcrl

attack_types = ["dos_data", "fuzzy_data", "spoof_gear_data", "spoof_rpm_data"]

timestr = time.strftime("%Y%m%d")
pathlib.Path("results/").mkdir(parents=True, exist_ok=True)
result_file = f"results/results_{timestr}.txt"
seed_value=42

for attack_type in attack_types:
    model = DCNN(seed_value)
    model.compile(Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy()])

    X_train, X_test, y_train, y_test = load_hcrl(attack_type)

    pathlib.Path("models/").mkdir(parents=True, exist_ok=True)
    filepath = "models/model_hcrl_"+attack_type+".h5"
    if not os.path.exists(filepath):
      checkpoint = ModelCheckpoint(filepath,monitor="loss",verbose=1, save_best_only=True, mode='auto')

      early = EarlyStopping(monitor="loss",
                            mode="auto",
                            patience=2, restore_best_weights=True)
      callbacks_list = [checkpoint, early]

      history = model.fit(X_train, y_train, epochs=30, verbose=1, callbacks=callbacks_list, batch_size=64)
    
    with open(result_file, "w") as res_fp:
        y_pred = model.predict(X_test)
        y_pred = y_pred > 0.5
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        res_fp.write(f"[{attack_type}]:\n\n")
        res_fp.write("Precision: "+str(precision)+"\n")
        res_fp.write("Recall:    "+str(recall)+"\n")
        res_fp.write("F1 score:  "+str(f1)+"\n")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        res_fp.write("FNR:       "+str(fn/(tp+fn))+"\n")
        res_fp.write("ER:        "+str((fp+fn)/(tp+tn+fp+fn))+"\n")
        res_fp.write(str(confusion_matrix(y_test, y_pred, labels=[0, 1]))+"\n\n")
