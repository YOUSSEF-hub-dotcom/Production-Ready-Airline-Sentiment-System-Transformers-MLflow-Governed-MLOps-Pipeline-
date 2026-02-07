# mlflow.py
import mlflow
import mlflow.pyfunc
import json
import torch
import pandas as pd
from sklearn.metrics import classification_report
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import logging

logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "DistilBERT-Sentiment-Analysis"
RUN_NAME = "DistilBERT_3Class_Sentiment"
MODEL_NAME = "DistilBERT_Airline_Sentiment"
ARTIFACT_PATH = "sentiment_model"


def run_mlflow_full_lifecycle(results):
    print("========== MLflow Full Lifecycle Started ==========")

    model = results["model"]
    tokenizer = results["tokenizer"]
    test_accuracy = results["test_accuracy"]
    preds = results["test_predictions"]
    labels = results["test_labels"]
    device = results["device"]
    params = results["params"]

    model_path = "distilbert_model"
    tokenizer_path = "distilbert_tokenizer"

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # TRACKING

    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"Tracking Run ID: {run_id}")

        mlflow.log_param("model", "distilbert-base-uncased")
        mlflow.log_param("device", str(device))
        mlflow.log_params(params)

        mlflow.log_metric("test_accuracy", test_accuracy)

        report = classification_report(
            labels,
            preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True
        )

        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)

        mlflow.log_artifact("classification_report.json")

        # PYFUNC MODEL & SIGNATURE


        class DistilBertPyFunc(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(context.artifacts["tokenizer"])
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    context.artifacts["model"]
                ).to(self.device)
                self.model.eval()
                logger.info(f"PyFunc Model loaded on: {self.device}")

            def predict(self, context, model_input):
                texts = model_input["text"].tolist() if isinstance(model_input, pd.DataFrame) else model_input["text"]
                logger.info(f"Inference requested for {len(model_input)} samples.")

                enc = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                label_map = {0: "negative", 1: "neutral", 2: "positive"}

                logger.info("Inference completed successfully.")

                return pd.DataFrame({
                    "label": [label_map[p.item()] for p in preds],
                    "confidence": [probs[i][preds[i]].item() for i in range(len(preds))]
                })

        input_example = pd.DataFrame({"text": ["The flight was delayed", "Excellent service!"]})

        model.eval()
        with torch.no_grad():
            dummy_enc = tokenizer(input_example["text"].tolist(), padding=True, truncation=True,
                                  return_tensors="pt").to(device)
            dummy_logits = model(**dummy_enc).logits
            dummy_preds = dummy_logits.argmax(dim=1)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        output_example = pd.DataFrame({
            "label": [label_map[p.item()] for p in dummy_preds],
            "confidence": [0.99] * len(dummy_preds)  # Dummy confidence
        })

        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            artifact_path=ARTIFACT_PATH,
            python_model=DistilBertPyFunc(),
            artifacts={"model": model_path, "tokenizer": tokenizer_path},
            input_example=input_example,
            signature=signature,
            pip_requirements=["torch", "transformers", "pandas", "numpy"]
        )

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"
    print(f"Registering model from: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    model_version = registered_model.version

    # TRANSITION → STAGING

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Staging"
    )

    # QUALITY GATE & PRODUCTION

    MIN_ACCURACY = 0.80
    MIN_MACRO_F1 = 0.75

    current_macro_f1 = report["macro avg"]["f1-score"]
    status = "Staging 🛑 (Low Performance)"

    if test_accuracy >= MIN_ACCURACY and current_macro_f1 >= MIN_MACRO_F1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"🚀 Model Promoted! Acc: {test_accuracy:.2f}, F1: {current_macro_f1:.2f}")

        status = "Production 🚀"
    else:
        logger.warning(f"🛑 Gate Failed! Required Acc > {MIN_ACCURACY}, got {test_accuracy:.2f}")

    logger.info(f"\n{'═' * 50}")
    logger.info(f"📦 Model: {MODEL_NAME}")
    logger.info(f"🔢 Version: {model_version}")
    logger.info(f"🏷️ Final Stage: {status}")
    logger.info(f"{'═' * 50}")

    return run_id
