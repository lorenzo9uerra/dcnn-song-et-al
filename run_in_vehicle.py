import time
import pathlib
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score
from keras.optimizers import Adam
from util.data import load_data_in_vehicle
from util.model import DCNN

car_brands = [
    'chevrolet',
    'hyundai',
    'kia'
]

timestr = time.strftime("%Y%m%d-%H%M%S")
pathlib.Path("results/").mkdir(parents=True, exist_ok=True)

seed_value = 42

with open(f"results/results_{timestr}_in_vehicle.txt", "w") as res_fp:
    results = {}
    results["seed"] = seed_value
    for car_brand in car_brands:
        results[car_brand] = {}
        for label in range(1,4):
            results[car_brand][label] = {}
            recalls = []
            precisions = []
            f1_scores = []
            for nrun in range(10):
                model = DCNN(seed_value)
                model.compile(Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=[keras.metrics.BinaryAccuracy()],)
                train_features, test_features, train_labels, test_labels = load_data_in_vehicle(car_brand, label)

                pathlib.Path("models/").mkdir(parents=True, exist_ok=True)
                filepath = f"models/model_{car_brand}_{label}.h5"
                checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="auto")

                early = EarlyStopping(monitor="loss", mode="auto", patience=2, restore_best_weights=True)
                callbacks_list = [checkpoint, early]

                history = model.fit(train_features,train_labels,epochs=30,verbose=1,callbacks=callbacks_list,batch_size=64)

                labels_pred = model.predict(test_features)
                labels_pred = labels_pred > 0.5
                recalls.append(recall_score(test_labels, labels_pred))
                f1_scores.append(f1_score(test_labels, labels_pred))
                precisions.append(precision_score(test_labels, labels_pred))

                # res_fp.write(f"\n{car_brand} for label={label}\n")
                # res_fp.write(classification_report(test_labels, labels_pred))
                # res_fp.write(str(confusion_matrix(test_labels, labels_pred)))

            results[car_brand][label]['recalls'] = recalls
            results[car_brand][label]['f1-scores'] = f1_scores
            results[car_brand][label]['precisions'] = precisions

    # write reults on file as python dictionary, ready to be parsed
    res_fp.write(str(results))