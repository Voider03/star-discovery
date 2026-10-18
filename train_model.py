"""
train_model.py - dual-branch (global + local view) residual 1D-CNN with a
vetting-feature branch, focal loss, and an ensemble of N models.
Train/validation split is BY STAR (synthetic copies stay with their source star).

    python train_model.py --processed processed --n_models 5 --outdir model
"""
import os, shutil, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight


def focal_loss(gamma=2.0, alpha=0.75):
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        at = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-at * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return fn


def res_block(x, filters, k=5, l2=1e-4):
    s = x
    x = layers.Conv1D(filters, k, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, k, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    if s.shape[-1] != filters:
        s = layers.Conv1D(filters, 1, padding="same")(s)
    return layers.Activation("relu")(layers.Add()([x, s]))


def branch(n, name, blocks):
    inp = layers.Input(shape=(n, 1), name=f"{name}_input")
    x, f = inp, 16
    for _ in range(blocks):
        x = layers.MaxPooling1D(2)(res_block(x, f))
        f *= 2
    return inp, layers.GlobalAveragePooling1D()(x)


def build_model(g_len, l_len, n_vet, dropout=0.4, l2=1e-4):
    gi, gf = branch(g_len, "global", 4)
    li, lf = branch(l_len, "local", 3)
    vi = layers.Input(shape=(n_vet,), name="vet_input")
    v = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(vi)
    v = layers.Dense(16, activation="relu")(layers.BatchNormalization()(v))
    x = layers.Concatenate()([gf, lf, v])
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(layers.BatchNormalization()(x))
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    m = models.Model([gi, li, vi], layers.Dense(1, activation="sigmoid")(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=focal_loss(),
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def threshold_for_precision(y, p, min_precision):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.5
    return float(th[ok[np.argmax(rc[:-1][ok])]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_models", type=int, default=5)
    ap.add_argument("--min_precision", type=float, default=0.9)
    ap.add_argument("--outdir", default="model")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    P = lambda n: np.load(os.path.join(a.processed, n))
    Xg, Xl, Xv, y = P("global_views.npy")[..., None], P("local_views.npy")[..., None], P("vet_features.npy"), P("y.npy")
    groups = pd.read_csv(os.path.join(a.processed, "star_manifest.csv"))["source"].str.split("::").str[0].values

    thresholds = []
    for seed in range(a.n_models):
        tf.keras.utils.set_random_seed(seed)
        tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xg, y, groups))
        cw = compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])
        m = build_model(Xg.shape[1], Xl.shape[1], Xv.shape[1])
        m.fit([Xg[tr], Xl[tr], Xv[tr]], y[tr], validation_data=([Xg[va], Xl[va], Xv[va]], y[va]),
              epochs=a.epochs, batch_size=a.batch_size, class_weight=dict(enumerate(cw)), verbose=2,
              callbacks=[callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10,
                                                 restore_best_weights=True)])
        prob = m.predict([Xg[va], Xl[va], Xv[va]], verbose=0).ravel()
        print(f"\n=== model {seed} ===")
        print(classification_report(y[va], (prob > 0.5).astype(int), zero_division=0))
        try:
            print("ROC-AUC:", round(roc_auc_score(y[va], prob), 4))
        except ValueError:
            pass
        thresholds.append(threshold_for_precision(y[va], prob, a.min_precision))
        m.compile(optimizer="adam", loss="binary_crossentropy")  # plain loss -> clean save/load
        m.save(os.path.join(a.outdir, f"model_{seed}.keras"))

    with open(os.path.join(a.outdir, "ensemble_config.txt"), "w") as f:
        f.write(f"n_models={a.n_models}\ndecision_threshold={float(np.mean(thresholds))}\n")
    for n in ["vet_mean.npy", "vet_std.npy"]:
        shutil.copy(os.path.join(a.processed, n), os.path.join(a.outdir, n))
    print(f"Saved {a.n_models}-model ensemble to {a.outdir}/")


if __name__ == "__main__":
    main()
