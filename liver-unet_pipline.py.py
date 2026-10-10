"""
Liver segmentation with U-Net: ONE script for everything.

  Step 1: LiTS NIfTI volumes  ->  PNG slices (images + masks)
  Step 2: Patient-wise split into train / val / test
  Step 3: Train U-Net
  Step 4: Test on the full test set (Dice, IoU) + save figures

How to run in PyCharm: open this file and press the green Run button.
Only change the settings in the CONFIG section below (mainly DATA_DIR).
"""
import glob
import os
import random
import re
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =====================================================================
# CONFIG  (change only this part)
# =====================================================================
DATA_DIR = r"D:\Program Files"                        # folder containing volume_pt1..pt5
SEG_DIR = os.path.join(DATA_DIR, "segmentations")     # folder containing segmentation-*.nii
OUTPUT_DATA = "Unet_data_final"                       # PNG dataset will be created here
RESULTS_DIR = "outputs"                               # models, curves, figures go here

REBUILD_DATASET = False   # True = convert again. After the first run set False to skip Step 1.

IMG_SIZE = 128           # must be divisible by 16
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MIN_LIVER_PIXELS = 300   # a slice is kept only if it has at least this many liver pixels
VAL_FRAC = 0.1           # fraction of PATIENTS used for validation
TEST_FRAC = 0.2          # fraction of PATIENTS used for testing
SEED = 42

HU_MIN, HU_MAX = -100, 400   # CT window for liver
EPS = 1e-7


# =====================================================================
# STEP 1 + 2: NIfTI -> PNG and patient-wise split
# =====================================================================
def find_pairs():
    """Return list of (case_id, volume_path, segmentation_path)."""
    vol_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "volume_pt*")))
    if not vol_dirs:
        raise FileNotFoundError(
            f"No 'volume_pt*' folders found inside: {DATA_DIR}\n"
            "Check DATA_DIR in the CONFIG section."
        )
    pairs = []
    for folder in vol_dirs:
        for f in sorted(os.listdir(folder)):
            if not (f.startswith("volume-") and f.endswith((".nii", ".nii.gz"))):
                continue
            case_id = f[len("volume-"):].replace(".nii.gz", "").replace(".nii", "")
            seg_path = None
            for ext in (".nii", ".nii.gz"):
                cand = os.path.join(SEG_DIR, f"segmentation-{case_id}{ext}")
                if os.path.exists(cand):
                    seg_path = cand
                    break
            if seg_path is None:
                print(f"[WARN] No segmentation found for {f}, skipping")
                continue
            pairs.append((case_id, os.path.join(folder, f), seg_path))
    if len(pairs) < 3:
        raise RuntimeError(
            f"Found only {len(pairs)} volume/segmentation pairs. Need at least 3.\n"
            f"Volumes searched in: {DATA_DIR}\\volume_pt*   Segmentations in: {SEG_DIR}"
        )
    return pairs


def split_by_patient(pairs):
    """Split by patient (NOT by slice) so there is no data leakage."""
    pairs = pairs[:]
    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    n_test = max(1, round(n * TEST_FRAC))
    n_val = max(1, round(n * VAL_FRAC))
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def save_volume_slices(case_id, v_path, s_path, out_img, out_mask):
    ct = nib.load(v_path).get_fdata(dtype=np.float32)
    seg = np.asanyarray(nib.load(s_path).dataobj).astype(np.uint8)

    if ct.shape != seg.shape:
        print(f"[WARN] Shape mismatch in case {case_id}: {ct.shape} vs {seg.shape}, skipping")
        return 0

    saved = 0
    for sl in range(ct.shape[2]):
        # LiTS labels: 1 = liver, 2 = tumor (tumor is inside liver) -> liver mask = label >= 1
        liver = seg[:, :, sl] >= 1
        if liver.sum() < MIN_LIVER_PIXELS:
            continue

        img = np.clip(ct[:, :, sl], HU_MIN, HU_MAX)
        img = ((img - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
        img = np.ascontiguousarray(np.rot90(img))
        mask = np.ascontiguousarray(np.rot90(liver.astype(np.uint8) * 255))

        name = f"case{case_id}_slice{sl:04d}.png"
        Image.fromarray(img).save(os.path.join(out_img, name))
        Image.fromarray(mask).save(os.path.join(out_mask, name))
        saved += 1
    return saved


def prepare_dataset():
    if os.path.exists(OUTPUT_DATA):
        if not REBUILD_DATASET:
            print(f"Step 1 skipped: '{OUTPUT_DATA}' already exists (REBUILD_DATASET = False).")
            return
        shutil.rmtree(OUTPUT_DATA)

    print("Step 1: Scanning volumes ...")
    pairs = find_pairs()
    train, val, test = split_by_patient(pairs)
    print(f"Patients -> total: {len(pairs)} | train: {len(train)} | val: {len(val)} | test: {len(test)}")

    print("Step 2: Converting to PNG (this takes a while) ...")
    for split_name, items in (("train", train), ("val", val), ("test", test)):
        out_img = os.path.join(OUTPUT_DATA, split_name, "images")
        out_mask = os.path.join(OUTPUT_DATA, split_name, "masks")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_mask, exist_ok=True)
        total = 0
        for i, (case_id, v_path, s_path) in enumerate(items, 1):
            n = save_volume_slices(case_id, v_path, s_path, out_img, out_mask)
            total += n
            print(f"  [{split_name}] {i}/{len(items)} case {case_id}: {n} slices")
        print(f"[{split_name}] total slices: {total}")
    print("Dataset conversion DONE.\n")


# =====================================================================
# Data loading
# =====================================================================
def load_split(split):
    """Load PNGs as uint8 arrays (saves RAM). Mask is resized with nearest-neighbour."""
    img_dir = os.path.join(OUTPUT_DATA, split, "images")
    mask_dir = os.path.join(OUTPUT_DATA, split, "masks")
    names = sorted(os.listdir(img_dir))
    if not names:
        raise RuntimeError(f"No images found in {img_dir}")

    X = np.zeros((len(names), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    Y = np.zeros((len(names), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    for i, name in enumerate(names):
        img = Image.open(os.path.join(img_dir, name)).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = Image.open(os.path.join(mask_dir, name)).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        X[i, :, :, 0] = np.asarray(img)
        Y[i, :, :, 0] = (np.asarray(mask) > 127).astype(np.uint8)
    return X, Y, names


def make_dataset(X, Y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =====================================================================
# Model
# =====================================================================
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    denom = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coef(y_true, y_pred))


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(size=IMG_SIZE, base=16, depth=4):
    assert size % (2 ** depth) == 0, f"IMG_SIZE must be divisible by {2 ** depth}"
    inputs = layers.Input((size, size, 1))
    x = inputs
    skips = []
    for i in range(depth):
        x = conv_block(x, base * 2 ** i)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

    x = conv_block(x, base * 2 ** depth)  # bottleneck

    for i in reversed(range(depth)):
        x = layers.UpSampling2D()(x)
        x = layers.concatenate([x, skips[i]])
        x = conv_block(x, base * 2 ** i)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="liver_unet")


# =====================================================================
# STEP 3: Training
# =====================================================================
def train():
    print("Step 3: Loading train / val data ...")
    X_train, Y_train, _ = load_split("train")
    X_val, Y_val, _ = load_split("val")
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    train_ds = make_dataset(X_train, Y_train, shuffle=True)
    val_ds = make_dataset(X_val, Y_val, shuffle=False)

    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )
    model.summary()

    best_path = os.path.join(RESULTS_DIR, "liver_unet_best.weights.h5")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_dice_coef", mode="max",
                        save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_dice_coef", mode="max", patience=8),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # go back to the best epoch (highest validation Dice)
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print(f"Loaded best weights from {best_path}")

    best_val_dice = max(history.history["val_dice_coef"])
    print(f"Best validation Dice: {best_val_dice * 100:.2f}%")

    try:
        model.save(os.path.join(RESULTS_DIR, "liver_unet_final.keras"))
    except Exception as e:  # saving the full model is optional
        print("[INFO] Full model could not be saved (weights are saved, this is fine):", e)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss (BCE + Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["dice_coef"], label="train")
    ax[1].plot(history.history["val_dice_coef"], label="val")
    ax[1].set_title("Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=200)
    plt.close(fig)

    del X_train, Y_train, X_val, Y_val, train_ds, val_ds
    return model


# =====================================================================
# STEP 4: Testing
# =====================================================================
def per_slice_metrics(y_true, y_pred):
    t = y_true.reshape(len(y_true), -1).astype(bool)
    p = y_pred.reshape(len(y_pred), -1).astype(bool)
    inter = np.sum(t & p, axis=1)
    dice = 2.0 * inter / (t.sum(1) + p.sum(1) + EPS)
    union = t.sum(1) + p.sum(1) - inter
    iou = inter / (union + EPS)
    return dice, iou


def save_figure(X, Y, P, indices, dice, path):
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(indices):
        axes[r, 0].imshow(X[i, :, :, 0], cmap="gray"); axes[r, 0].set_title("CT Image")
        axes[r, 1].imshow(Y[i, :, :, 0], cmap="gray"); axes[r, 1].set_title("Ground Truth")
        axes[r, 2].imshow(P[i] > 0.5, cmap="gray"); axes[r, 2].set_title(f"Prediction (Dice {dice[i]:.3f})")
        for c in range(3):
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def evaluate(model):
    print("\nStep 4: Testing on the full test set ...")
    X, Y, names = load_split("test")
    print(f"Test slices: {len(X)}")

    test_ds = tf.data.Dataset.from_tensor_slices(X).map(
        lambda x: tf.cast(x, tf.float32) / 255.0
    ).batch(16)
    P = model.predict(test_ds, verbose=1)[:, :, :, 0]

    T = Y[:, :, :, 0] > 0
    Pb = P > 0.5
    dice, iou = per_slice_metrics(T, Pb)

    # patient-level Dice: all slices of one patient pooled together
    groups = defaultdict(list)
    for i, n in enumerate(names):
        m = re.match(r"case(.+?)_slice", n)
        if m:
            groups[m.group(1)].append(i)
    case_dice = []
    for idx in groups.values():
        t, pr = T[idx], Pb[idx]
        case_dice.append(2.0 * np.sum(t & pr) / (t.sum() + pr.sum() + EPS))

    lines = [
        "========== TEST RESULTS ==========",
        f"Test slices        : {len(X)}",
        f"Test patients      : {len(groups)}",
        f"Slice-level Dice   : {dice.mean() * 100:.2f}% (+/- {dice.std() * 100:.2f})",
        f"Slice-level IoU    : {iou.mean() * 100:.2f}% (+/- {iou.std() * 100:.2f})",
        f"Patient-level Dice : {np.mean(case_dice) * 100:.2f}% (+/- {np.std(case_dice) * 100:.2f})",
        "==================================",
    ]
    print("\n" + "\n".join(lines))
    with open(os.path.join(RESULTS_DIR, "test_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    rng = np.random.default_rng(SEED)
    random_idx = rng.choice(len(X), size=min(3, len(X)), replace=False)
    worst_idx = np.argsort(dice)[:3]
    save_figure(X, Y, P, random_idx, dice, os.path.join(RESULTS_DIR, "RESULT_random_examples.png"))
    save_figure(X, Y, P, worst_idx, dice, os.path.join(RESULTS_DIR, "RESULT_worst_examples.png"))
    print(f"Results saved in the '{RESULTS_DIR}' folder.")


# =====================================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("TensorFlow:", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))

    prepare_dataset()      # Step 1 + 2
    model = train()        # Step 3
    evaluate(model)        # Step 4
    print("\nALL DONE.")


if __name__ == "__main__":
    main()