import argparse
import os
from dataclasses import asdict, dataclass, field
from getpass import getpass

import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from clearml import Task
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)
from sklearn.model_selection import train_test_split


@dataclass
class CFG:
    seed: int = 777
    project_name: str = "Машинное обучение с помощью ClearML и Pytorch Lighting"
    experiment_name: str = "Lab 1 Catboost model logging"
    tags: list = field(default_factory=lambda: ['Lab1', 'Catboost'])
    catboost_params: dict = field(default_factory=lambda: {
        'depth': 4,
        'learning_rate': 0.06,
        'loss_function': "MultiClass",
        'custom_metric': ["Recall"],
        'cat_features': ["model", "car_type", "fuel_type"],
        'colsample_bylevel': 0.098,
        'subsample': 0.95,
        'l2_leaf_reg': 9,
        'min_data_in_leaf': 243,
        'max_bin': 187,
        'random_strength': 1,
        'task_type': "CPU",
        'thread_count': -1,
        'bootstrap_type': "Bernoulli",
        'random_seed': 777,
        'early_stopping_rounds': 50
})


def check_clearml_env():
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]

    for var in [var for var in required_env_vars if os.getenv(var) is None]:
            os.environ[var] = getpass(f"Enter {var}: ")
    else:
        print("All environment variables are set.")

def get_data(link):
    data = pd.read_csv(link)
    print("Data has been downloaded.")
    return data


def process_data(df, cat_features, targets, features2drop):
    try:
        processed_df = df.copy()
        filtered_features = [i for i in processed_df.columns if (i not in targets and i not in features2drop)]
        print("cat_features", cat_features)
        print("num_features", len([i for i in filtered_features if i not in cat_features]))
        print("targets", targets)

        for c in cat_features:
            processed_df[c] = processed_df[c].astype(str)

        print("Data processing is finished.")
        return processed_df, filtered_features
    except Exception as e:
        print(e)


def get_training_datasets(df, X, target):
    try:
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        X_train = train[X]
        y_train = train[target]
        X_test = test[X]
        y_test = test[target]

        print("Datasets for training are ready to go.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(e)


def train_catboost(X_train, y_train, X_test, y_test, config, need_verbose=False):
    model = CatBoostClassifier(**config)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=need_verbose)
    model.save_model("cb_model.cbm")
    print("Model has been trained and saved.")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    cls_report = classification_report(
        y_test, y_pred,
        target_names=[str(cls) for cls in sorted(y_test.unique())],
        output_dict=True)
    cls_report = pd.DataFrame(cls_report).T
    return acc, f1, recall, cls_report



def parse_args():
    parser = argparse.ArgumentParser(description="CLI Catboost Training")
    parser.add_argument(
        "-v", "--verbose",
        type=lambda x: int(x) if x and x.isdigit() else False,
        default=False,
        nargs="?",
        help="Verbose mode: False (default) or an integer value"
    )
    parser.add_argument("-it", "--iterations", type=int, default=500, help="Number of iterations (default: 500)")
    return parser.parse_args()


if __name__ == '__main__':
    check_clearml_env()
    configuration = asdict(CFG())
    args = parse_args()
    print(f"Verbose: {args.verbose}")
    print(f"Iterations: {args.iterations}")
    configuration['catboost_params']['iterations'] = args.iterations
    np.random.seed(configuration['seed'])

    task = Task.init(
        project_name=configuration['project_name'],
        task_name=configuration['experiment_name'],
        tags=configuration['tags'],
    )
    logger = task.get_logger()

    # SAVE MODEL'S HYPOS
    task.connect(mutable=configuration['catboost_params'], name='Catboost Training Parameters')

    # DATASET DOWNLOADING
    df, X = process_data(
        df=get_data(link='https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv'),
        cat_features=configuration['catboost_params']['cat_features'],
        targets=["target_class", "target_reg"],
        features2drop=["car_id"]
    )

    # TRAIN-TEST SPLIT
    X_train, y_train, X_test, y_test = get_training_datasets(df, X, "target_class")

    # LOGG VALIDATION DATASET
    validation_dataset_for_logging = pd.concat([X_test, y_test.to_frame()], axis=1)
    logger.report_table(
        title="Validation",
        series="Dataset",
        table_plot=validation_dataset_for_logging.reset_index(),
    )

    plt.figure(figsize=(8, 5))
    sns.countplot(y=y_train)
    plt.title("Target Distribution on Train")
    plt.show()
    plt.close()

    logger.report_table(
        title='Validation',
        series='Missing Values',
        table_plot=X_train.isna().sum().reset_index(name="missing_count")
    )

    # TRAIN & SAVE MODEL
    model = train_catboost(
        X_train, y_train, X_test, y_test,
        config=configuration['catboost_params'],
        need_verbose=args.verbose
    )

    # SAVE METRICS
    acc, f1, recall, cls_report = evaluate_model(model, X_test, y_test)
    print("Evaluation process is finished.")

    logger.report_scalar(title='Metrics on Validation', series='Accuracy', value=acc, iteration=0)
    logger.report_scalar(title='Metrics on Validation', series='F1 Score', value=f1, iteration=0)
    logger.report_scalar(title='Metrics on Validation', series='Recall', value=recall, iteration=0)
    logger.report_table(
        title="Training Results",
        series="Classification Report",
        table_plot=cls_report.reset_index(),
    )
    print("Process is finished successfully.")
    task.mark_completed()
    task.close()
