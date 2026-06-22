import argparse, math, os, random
from collections import defaultdict, Counter
import pandas as pd

def infer_split(path: str) -> str:
    p = path.replace("\\", "/")
    # suppose /processed/<split>/<class>/...
    parts = p.split("/")
    if "processed" in parts:
        i = parts.index("processed")
        if i+1 < len(parts):
            return parts[i+1]
    # fallback: chercher /train/ /val/ /test/
    for s in ("train","val","test"):
        if f"/{s}/" in p:
            return s
    return "unknown"

def stratified_sample(df: pd.DataFrame, n_target: int, label_col="label", seed=42):
    """Essaie d'équilibrer par classe (au plus possible)."""
    random.seed(seed)
    labels = sorted(df[label_col].unique())
    k = len(labels)
    # quota de base égalitaire par classe
    base = n_target // k
    remainder = n_target - base * k
    # répartir le reste sur les premières classes
    per_class_target = {lbl: base + (1 if i < remainder else 0) for i, lbl in enumerate(labels)}

    rows = []
    for lbl in labels:
        sub = df[df[label_col]==lbl]
        if len(sub) <= per_class_target[lbl]:
            rows.append(sub)  # pas assez: on prend tout
        else:
            rows.append(sub.sample(per_class_target[lbl], random_state=seed))
    out = pd.concat(rows, ignore_index=True)

    # si encore insuffisant (classes trop petites), on complète proportionnellement
    if len(out) < n_target:
        missing = n_target - len(out)
        remaining = df.drop(out.index)
        if len(remaining) > 0:
            add = remaining.sample(min(missing, len(remaining)), random_state=seed)
            out = pd.concat([out, add], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="data/interim/dataset_processed.csv")
    ap.add_argument("--out-csv", default="data/interim/dataset_subset.csv")
    ap.add_argument("--train-size", type=int, default=1500)
    ap.add_argument("--split-train", type=float, default=0.7)
    ap.add_argument("--split-val", type=float, default=0.2)
    ap.add_argument("--split-test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    # ajoute une colonne split à partir du chemin
    df["split"] = df["filepath"].apply(infer_split)

    # calcule les tailles cibles en gardant le ratio
    total_needed = int(round(args.train_size / args.split_train))
    val_needed = int(round(total_needed * args.split_val))
    test_needed = total_needed - args.train_size - val_needed

    print(f"Target sizes -> total={total_needed}, train={args.train_size}, val={val_needed}, test={test_needed}")

    out_parts = []
    # TRAIN (équilibré par classe autant que possible)
    train_df = df[df["split"]=="train"]
    train_sub = stratified_sample(train_df, args.train_size, seed=args.seed)
    out_parts.append(train_sub)

    # VAL / TEST (on garde la distribution naturelle, mais on limite en taille)
    for split_name, n_target in [("val", val_needed), ("test", test_needed)]:
        part = df[df["split"]==split_name]
        if len(part) <= n_target:
            out_parts.append(part)
        else:
            # Sampling proportionnel par classe
            rows = []
            for lbl, sub in part.groupby("label"):
                k = len(part)
                want = int(round(n_target * len(sub) / k))
                rows.append(sub.sample(min(want, len(sub)), random_state=args.seed))
            sub_df = pd.concat(rows, ignore_index=True)
            # Ajustement si arrondis
            if len(sub_df) > n_target:
                sub_df = sub_df.sample(n_target, random_state=args.seed)
            elif len(sub_df) < n_target:
                extra = part.drop(sub_df.index).sample(min(n_target-len(sub_df), len(part)-len(sub_df)),
                                                       random_state=args.seed)
                sub_df = pd.concat([sub_df, extra], ignore_index=True)
            out_parts.append(sub_df)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # stats
    print("Final sizes:", out["split"].value_counts().to_dict())
    print("Train per class:", Counter(out[out["split"]=="train"]["label"]))
    print("Val per class:", Counter(out[out["split"]=="val"]["label"]))
    print("Test per class:", Counter(out[out["split"]=="test"]["label"]))

    out[["filepath","label"]].to_csv(args.out_csv, index=False)
    print(f"✅ Wrote subset CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
