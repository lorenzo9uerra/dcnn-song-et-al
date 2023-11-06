import time
import pathlib
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
)
from keras.optimizers import Adam
from util.data import load_data_road
from util.model import DCNN

attack_types = [
    "fuzzing_attack",
    "correlated_signal_attack",
    "correlated_signal_attack_masquerade",
    "max_engine_coolant_temp_attack",
    "max_engine_coolant_temp_attack_masquerade",
    "max_speedometer_attack",
    "max_speedometer_attack_masquerade",
    "reverse_light_off_attack",
    "reverse_light_off_attack_masquerade",
    "reverse_light_on_attack",
    "reverse_light_on_attack_masquerade",
]
timestr = time.strftime("%Y%m%d-%H%M%S")
pathlib.Path("results/").mkdir(parents=True, exist_ok=True)

seed_value = 42

# total number of runs per each attack
runs_number = 10

with open(f"results/results_{timestr}_road.txt", "w") as res_fp:
    results = {}
    results["seed"] = seed_value
    for attack_type in attack_types:
        recalls = []
        precisions = []
        f1_scores = []
        results[attack_type] = {}
        for nrun in range(runs_number):
            model = DCNN(seed_value)
            model.compile(
                Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=[keras.metrics.BinaryAccuracy()],
            )
            train_features, test_features, train_labels, test_labels = load_data_road(attack_type)

            pathlib.Path("models/").mkdir(parents=True, exist_ok=True)
            filepath = f"models/model_{attack_type}.h5"
            checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="auto")

            early = EarlyStopping(monitor="loss", mode="auto", patience=2, restore_best_weights=True)
            callbacks_list = [checkpoint, early]

            history = model.fit(
                train_features,
                train_labels,
                epochs=30,
                verbose=1,
                callbacks=callbacks_list,
                batch_size=64,
            )
            labels_pred = model.predict(test_features)
            labels_pred = labels_pred > 0.5
            recalls.append(recall_score(test_labels, labels_pred))
            f1_scores.append(f1_score(test_labels, labels_pred))
            precisions.append(precision_score(test_labels, labels_pred))

            # res_fp.write(f"\n{attack_type}\n")
            # res_fp.write(classification_report(test_labels, labels_pred))
            # res_fp.write(str(confusion_matrix(test_labels, labels_pred)))
            # np.save(f"../results/preds/{attack_type}_{nrun}.npy", labels_pred)

        results[attack_type]["precisions"] = precisions
        results[attack_type]["recalls"] = recalls
        results[attack_type]["f1-scores"] = f1_scores

    # write reults on file as python dictionary, ready to be parsed
    res_fp.write(str(results))
