from data_utils import load_and_clean_data, make_splits
from features import (
    build_spec_feature_matrix,
    build_text_feature_matrix,
)
from models import train_decision_tree, train_knn, train_naive_bayes_text, tune_decision_tree
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#This is the main file that gets executed from the terminal when we run "python main.py"

# Helper function: compute and print only validation accuracy for a model 
# Called after training each model to compare
# how well they do on the valudation set.
def simple_val_accuracy(model, X_val, y_val, name):
    """Print only validation accuracy for cleaner output."""
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print(f"{name} (VAL) accuracy = {acc:.3f}")
    return acc

#Evaluate choesen model on held-out Test split
#Prints: overall accuracy, macro-F1, and confusion matrix (how ofen we confuse each band)
def evaluate_test(model, X_test, y_test, name):
    """Evaluate the chosen model on the test split."""
    print(f"\n=== TEST EVALUATION: {name} ===")
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}")
    print("Confusion matrix:")
    print(cm)


def main():
    # 1.Load + split
    df = load_and_clean_data("data.csv", max_rows=100_000)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = make_splits(df)

    # 2. SPEC-ONLY matrices
    print("\n--- Building SPEC feature matrix ---")
    X_train_spec, X_val_spec, X_test_spec, preproc = build_spec_feature_matrix(
        X_train_df, X_val_df, X_test_df
    )

    # 3. TEXT-ONLY matrices
    print("\n--- Building TEXT feature matrix ---")
    X_train_text, X_val_text, X_test_text, vec = build_text_feature_matrix(
        X_train_df, X_val_df, X_test_df
    )

    # 4.Train all models (same as before)
    print("\nTuning Decision Tree (spec-only)...")
    tree_spec = tune_decision_tree(
        X_train_spec, y_train, X_val_spec, y_val, name="Decision Tree (spec-only)"
    )
    acc_tree_spec = simple_val_accuracy(tree_spec, X_val_spec, y_val, "Decision Tree (spec-only)")

    #k-NN on SPEC-ONLY
    print("\nTraining k-NN (spec-only)...")
    knn_spec = train_knn(X_train_spec, y_train, X_val_spec, y_val)
    acc_knn_spec = simple_val_accuracy(knn_spec, X_val_spec, y_val, "k-NN (spec-only)")

    #Naive Bayes on TEXT-ONLY
    print("\nTraining Naive Bayes (text-only)...")
    nb_text = train_naive_bayes_text(X_train_text, y_train, X_val_text, y_val)
    acc_nb_text = simple_val_accuracy(nb_text, X_val_text, y_val, "Naive Bayes (text-only)")

    # 5. Build fused SPEC + TEXT matrix and tune Decision Tree on it
    print("\nBuilding fused SPEC + TEXT matrix...")
    X_train_fused = sparse.hstack([X_train_spec, X_train_text])
    X_val_fused = sparse.hstack([X_val_spec, X_val_text])
    X_test_fused = sparse.hstack([X_test_spec, X_test_text])

    print("\nTuning Decision Tree (fused spec+text)...")
    tree_fused = tune_decision_tree(
        X_train_fused, y_train, X_val_fused, y_val, name="Decision Tree (fused)"
    )
    acc_tree_fused = simple_val_accuracy(tree_fused, X_val_fused, y_val, "Decision Tree (fused)")

    # 6. Print validation summary so we can compare models

    print("\n=== VALIDATION SUMMARY ===")
    print(f"1) Decision Tree (spec-only): {acc_tree_spec:.3f}")
    print(f"2) k-NN (spec-only):          {acc_knn_spec:.3f}")
    print(f"3) Naive Bayes (text-only):   {acc_nb_text:.3f}")
    print(f"4) Decision Tree (fused):      {acc_tree_fused:.3f}")

    #Store models + their test feature sets in a list to make selection easy
    model_list = [
        ("Decision Tree (spec-only)", tree_spec, X_test_spec),
        ("k-NN (spec-only)", knn_spec, X_test_spec),
        ("Naive Bayes (text-only)", nb_text, X_test_text),
        ("Decision Tree (fused)", tree_fused, X_test_fused),
    ]

    #menu printed to terminal 
    print("\nWhich model would you like to TEST on the held-out 20%?")
    print("1 = Decision Tree (spec-only)")
    print("2 = k-NN (spec-only)")
    print("3 = Naive Bayes (text-only)")
    print("4 = Decision Tree (fused)")

    choice = input("Enter model number (1-4): ").strip()
    while choice not in {"1", "2", "3", "4"}:
        choice = input("Invalid choice. Enter 1,2,3,4: ").strip()

    idx = int(choice) - 1
    model_name, model_obj, X_test_features = model_list[idx]

    #7.Final test evaluation for the selected model
    evaluate_test(model_obj, X_test_features, y_test, model_name)

if __name__ == "__main__":
    main()