# model.py

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nltk
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def build_and_train_distilbert_model(df, epochs, lr, batch_size):
    logger.info("===========================>>>> Deep Learning (DistilBERT) Model")
    logger.info(f"Params: Epochs={epochs}, LR={lr}, BatchSize={batch_size}")

    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    logger.info("Mapping sentiments to numerical labels...")
    df['label'] = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

    df = df[['cleaned_text', 'label']].copy()
    logger.info("Dataset after mapping labels:")
    print(df.head())

    logger.info("\nBalancing dataset using Synonym Augmentation...")
    max_count = df['label'].value_counts().max()
    aug = naw.SynonymAug(aug_src='wordnet')

    augmented_texts, augmented_labels = [], []
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        current_count = len(subset)
        needed = max_count - current_count

        augmented_texts.extend(subset['cleaned_text'].tolist())
        augmented_labels.extend(subset['label'].tolist())

        if needed > 0:
            generated_texts = set()
            while len(generated_texts) < needed:
                row = subset.sample(1).iloc[0]
                new_text = aug.augment(row['cleaned_text'])
                if isinstance(new_text, list):
                    new_text = " ".join(new_text)
                if new_text not in generated_texts and new_text not in subset['cleaned_text'].values:
                    generated_texts.add(new_text)

            augmented_texts.extend(list(generated_texts))
            augmented_labels.extend([label] * len(generated_texts))

    balanced_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    logger.info("Balanced labels after augmentation:")
    print(balanced_df['label'].value_counts())

    logger.info("\nSplitting data into train and test...")
    train_df, test_df = train_test_split(
        balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label']
    )

    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    logger.info("Tokenizing training and test data...")
    train_encodings = tokenizer(
        list(train_df['text']),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    test_encodings = tokenizer(
        list(test_df['text']),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    train_labels = torch.tensor(train_df['label'].values)
    test_labels = torch.tensor(test_df['label'].values)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model Training running on: {device}")
    logger.info("Loading DistilBERT model for 3-class classification...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    num_epochs = epochs
    num_training_steps = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), desc="Training")

    # 1. تصحيح مكان الـ Loss Logging (لازم يكون جوه الـ training loop)
    logger.info("\nStarting training...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.update(1)

            # تسجيل الـ Loss كل 50 خطوة (مكانها الصحيح هنا)
            if i % 50 == 0:
                logger.info(f"Epoch {epoch + 1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed - Average Loss: {avg_loss:.4f}")

    progress_bar.close()

    # 2. إضافة تتبع الأخطاء (Miss-classification) جوه الـ Evaluation
    logger.info("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

            # هنا بنلف على الـ Batch ونشوف الغلطات
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if p != l:
                    # بنستخدم debug عشان الـ logs ماتبقاش زحمة جداً
                    logger.debug(f"Miss-classification: Predicted {p} but was {l}")

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive']
    ))

    logger.info("===========================>>>> DistilBERT Model Training Completed!")

    return {
        'model': model,
        'tokenizer': tokenizer,
        'test_accuracy': accuracy,
        'test_predictions': all_preds,
        'test_labels': all_labels,
        'device': device,
        'params': {'epochs': epochs, 'lr': lr, 'batch_size': batch_size}
    }
