import os, argparse, joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    mlflow.set_experiment("ml-lifecycle-azureml")
    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=args.max_iter)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_metric("accuracy", float(acc))

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        # Optional: save the model file into the MLflow run folder too
        # mlflow.log_artifact("models/model.joblib")

        print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
