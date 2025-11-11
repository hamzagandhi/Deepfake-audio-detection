
print("🚀 Training script started!")

# train_asvspoof_la.py
import os
import argparse
import glob
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from sklearn.metrics import accuracy_score, f1_score, roc_curve

# -------------------------
# Utilities
# -------------------------
def find_protocol_file(protocols_dir, split_keyword):
    # protocol filenames vary, so pick the one containing split_keyword and 'cm'
    for p in Path(protocols_dir).glob("*"):
        name = p.name.lower()
        if "cm" in name and split_keyword in name:
            return str(p)
    return None

def parse_cm_protocol(protocol_path, audio_root):
    """
    Parse ASVspoof CM protocol files.
    Heuristic parser: expects lines with at least [utt_id ... label] where label is 'bonafide' or 'spoof'
    Many protocol formats: columns could be [utt_id, speaker, _, _, label] or [utt_id label].
    We'll try to detect the filename token (usually second token) and the label (last token).
    Returns list of (wav_path, int_label) where int_label: 0 = bonafide (real), 1 = spoof (fake).
    """
    pairs = []
    with open(protocol_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            # Heuristic: last token often label
            label_token = toks[-1].lower()
            if label_token in ("bonafide", "genuine", "real", "target"):
                label = 0
            elif label_token in ("spoof", "spoofed", "fake", "impersonation", "attack"):
                label = 1
            else:
                # fallback: look for 'bonafide'/'spoof' anywhere
                label = None
                for t in toks:
                    tt = t.lower()
                    if tt in ("bonafide", "genuine", "real", "target"):
                        label = 0
                        break
                    if tt in ("spoof", "fake", "spoofed", "attack"):
                        label = 1
                        break
                if label is None:
                    # can't determine label; skip
                    continue

            # find token that looks like audio id (contains 'LA' or startswith 'ASV' or contains '.flac' / file-like)
            audio_token = None
            for t in toks:
                if t.lower().endswith(".flac") or t.lower().endswith(".wav"):
                    audio_token = t
                    break
            if audio_token is None:
                # try tokens with 'LA' or 'LA_' or having digits and '_' patterns
                for t in toks:
                    if "la" in t.lower() and "_" in t:
                        audio_token = t
                        break
                if audio_token is None:
                    # as last resort take first token
                    audio_token = toks[0]

            # Build path: search for file under audio_root matching audio_token (without extension)
            # Many protocol audio_tokens are just utterance IDs without extension. We'll search for matching file.
            base = audio_token.split(".")[0]
            # Search with glob (may be slower but robust)
            matches = list(Path(audio_root).rglob(f"{base}.*"))
            chosen = None
            if matches:
                # pick first existing .flac/.wav
                for m in matches:
                    if m.suffix.lower() in (".flac", ".wav"):
                        chosen = m
                        break
            if chosen is None:
                # try direct join (audio_root/base.flac)
                guess = Path(audio_root) / (base + ".flac")
                if guess.exists():
                    chosen = guess
                else:
                    guess2 = Path(audio_root) / (base + ".wav")
                    if guess2.exists():
                        chosen = guess2
            if chosen and chosen.exists():
                pairs.append((str(chosen), label))
            else:
                # not found; skip
                continue
    return pairs

def load_split_pairs(asv_root, split_name):
    """
    asv_root: root path containing ASVspoof2019 LA folders (train/dev/eval + protocols)
    split_name: 'train', 'dev', or 'eval'
    """
    # locate CM protocol folder (common name ASVspoof2019_LA_cm_protocols or similar)
    protocol_dir = None
    for candidate in Path(asv_root).glob("*cm*protocol*"):
        if candidate.is_dir():
            protocol_dir = str(candidate)
            break
    if protocol_dir is None:
        # try sibling directories
        for candidate in Path(asv_root).iterdir():
            if candidate.is_dir() and "protocol" in candidate.name.lower():
                protocol_dir = str(candidate)
                break
    if protocol_dir is None:
        raise FileNotFoundError("Could not find CM protocol folder under ASV root: " + asv_root)

    # find relevant protocol file
    proto_file = find_protocol_file(protocol_dir, split_name)
    if proto_file is None:
        raise FileNotFoundError(f"Could not find CM protocol file for split {split_name} in {protocol_dir}")

    # audio folder for the split: pattern may be ASVspoof2019_LA_train/flac or similar
    audio_folder = None
    # possible folder names
    for p in Path(asv_root).glob("*" + split_name + "*"):
        if p.is_dir():
            # find nested folder containing flac/wav files
            for candidate in p.iterdir():
                if candidate.is_dir():
                    # check if contains audio files
                    if any(candidate.glob("*.flac")) or any(candidate.glob("*.wav")):
                        audio_folder = str(candidate)
                        break
            if audio_folder:
                break
    if audio_folder is None:
        # fallback: search for any flac in subtree with split_name in path
        matches = list(Path(asv_root).rglob(f"*{split_name}*/*.flac"))
        if matches:
            audio_folder = str(matches[0].parent)
    if audio_folder is None:
        # as last resort, use asv_root itself
        audio_folder = str(asv_root)

    pairs = parse_cm_protocol(proto_file, audio_folder)
    return pairs

def build_dataset_from_pairs(pairs, max_examples=None):
    # pairs: list of (path,label)
    if max_examples:
        pairs = pairs[:max_examples]
    records = {"path": [], "label": []}
    for p,l in pairs:
        records["path"].append(p)
        records["label"].append(int(l))
    return Dataset.from_dict(records)

# Simple audio loader used in preprocessing step (librosa)
def load_and_resample(path, target_sr=16000):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # normalize
    maxv = np.max(np.abs(audio))
    if maxv > 0:
        audio = audio / maxv
    return audio

# -------------------------
# EER function
# -------------------------
def compute_eer(labels, scores):
    # labels: 0 = bonafide, 1 = spoof
    # scores: higher -> more likely spoof (we'll assume model outputs P(spoof))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # find threshold where FPR and FNR are closest
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

# -------------------------
# Main training pipeline
# -------------------------
def main(args):
    asv_root = args.asv_root
    model_id = args.base_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build train/dev pairs
    print("Preparing datasets (this may take a few minutes)...")
    train_pairs = load_split_pairs(asv_root, "train")
    dev_pairs   = load_split_pairs(asv_root, "dev")
    eval_pairs  = load_split_pairs(asv_root, "eval")

    print(f"Found {len(train_pairs)} train, {len(dev_pairs)} dev, {len(eval_pairs)} eval pairs.")

    # convert to HF Dataset
    train_ds = build_dataset_from_pairs(train_pairs)
    dev_ds = build_dataset_from_pairs(dev_pairs)
    eval_ds = build_dataset_from_pairs(eval_pairs)

    # Load feature extractor & model
    print("Loading model:", model_id)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    # Optionally freeze encoder
    if args.freeze_encoder:
        for p in model.wav2vec2.parameters():
            p.requires_grad = False
        print("Frozen wav2vec2 encoder parameters.")

    model.to(device)

    # Preprocessing function for dataset.map
    def preprocess_function(batch):
        path = batch["path"]
        audio = load_and_resample(path, target_sr=16000)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        # return numpy arrays or lists to be stored in dataset
        input_values = inputs["input_values"].squeeze(0).numpy()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0).numpy()
        else:
            attention_mask = np.ones_like(input_values, dtype=int)
        return {"input_values": input_values, "attention_mask": attention_mask, "label": batch["label"]}

    print("Preprocessing datasets (this may take time and RAM)...")
    # map with batched=False to process each file
    train_ds = train_ds.map(preprocess_function, remove_columns=["path", "label"])
    dev_ds = dev_ds.map(preprocess_function, remove_columns=["path", "label"])
    eval_ds = eval_ds.map(preprocess_function, remove_columns=["path", "label"])

    # Convert fields to correct formats expected by Trainer (numpy -> lists)
    def collate_fn(batch):
        # batch is list of dicts with input_values (np arrays) and attention_mask
        input_values = [torch.tensor(x["input_values"], dtype=torch.float32) for x in batch]
        attention_masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        # pad to longest
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
        return {"input_values": input_values, "attention_mask": attention_masks, "labels": labels}

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_eer",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=3
    )

    # Custom compute metrics
    def compute_metrics(pred):
        logits = pred.predictions
        labels = pred.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        spoof_probs = probs[:, 1]  # p(spoof)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        eer = compute_eer(labels, spoof_probs)
        return {"accuracy": acc, "f1": f1, "eer": eer}

    # Wrap datasets with torch-friendly format using map to set numpy dtypes
    # datasets already have input_values arrays; we will use data collator by passing collate_fn to Trainer via data_collator
    train_dataset = train_ds
    eval_dataset = dev_ds

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=feature_extractor,  # feature_extractor used as tokenizer for HF Trainer compatibility
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate on eval set (final test)
    print("Evaluating on evaluation set...")
    # Use Trainer.predict to get probabilities on eval set
    raw_pred = trainer.predict(eval_ds)
    logits = raw_pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    spoof_probs = probs[:,1]
    labels = raw_pred.label_ids
    eer_eval = compute_eer(labels, spoof_probs)
    print(f"Eval EER: {eer_eval:.4f}")

    # Save model
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print("Saved fine-tuned model to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_root", type=str, required=True,
                        help="Root folder of ASVspoof2019 LA (e.g., C:/Users/SHABBIR/Downloads/ASVspoof2019/LA/LA)")
    parser.add_argument("--base_model", type=str, default="Zeyadd-Mostaffa/Deepfake-Audio-Detection-v1",
                        help="Hugging Face model id (Wav2Vec2-based)")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_asvspoof_la")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze wav2vec2 encoder (train head only)")
    args = parser.parse_args()
    main(args)
