from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# This file contains all the machine learning models we use:
#   - Decision Tree (spec-only and fused)
#   - k-Nearest Neighbours (spec-only)
#   - Multinomial Naive Bayes (text-only)

#Train a single Decision Tree with fixed hyperparameters
def train_decision_tree(X_train, y_train, X_val, y_val):

    #Kept for debugging or quick runs; not used in the tuned pipeline.
    clf = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=50,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    #evaluate the valudation split
    y_val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"DecisionTree (val): acc={acc:.3f}, macro-F1={macro_f1:.3f}")
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    #return trained tree
    return clf


#Do small hyperparameter seach for a Decision Tree
def tune_decision_tree(X_train, y_train, X_val, y_val, name: str = "Decision Tree"):
    
    # We try a few combiniations of max_depth and min_samples
    # For each combo: 
    #  - train on X_train, y_train
    #  - evaluate on X_val, y_val
    #  - print accuracy and macro-F1
    # Then pick the model with thebest validation macro-F1 and return

    #grid of hyperparams to try
    param_grid = {
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [20, 50, 100],
    }

    best_clf = None
    best_params = None
    best_f1 = -1.0      #start lower than any real F1

    print(f"\n--- Hyperparameter search for {name} ---")
    for max_depth in param_grid["max_depth"]:
        for min_leaf in param_grid["min_samples_leaf"]:
            # Create a tree with this specific (max_depth, min_samples_leaf)
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=42,
            )
            # Train on training split
            clf.fit(X_train, y_train)
            # Validate on validation split
            y_val_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            macro_f1 = f1_score(y_val, y_val_pred, average="macro")

            print(
                f"[{name}] max_depth={max_depth}, "
                f"min_samples_leaf={min_leaf} -> "
                f"acc={acc:.3f}, macro-F1={macro_f1:.3f}"
            )

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_clf = clf
                best_params = (max_depth, min_leaf)

    # Print the best hyperparameters we found
    print(
        f"\nBest {name} params: max_depth={best_params[0]}, "
        f"min_samples_leaf={best_params[1]}, macro-F1={best_f1:.3f}"
    )

    #Print final validation performance for the chosen model
    y_val_pred = best_clf.predict(X_val)
    print(f"\n{name} (VAL) with best params:")
    print("Accuracy:", f"{accuracy_score(y_val, y_val_pred):.3f}")
    print("Macro-F1:", f"{f1_score(y_val, y_val_pred, average='macro'):.3f}")
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    # Return the *best* trained Decision Tree to main.py
    return best_clf


#Train a K-Nearest Neighbours classifier on the SPEC features
#   - To classify a new listing, k-NN looks at the 'k' closest listings
#     in the training data (based on feature distance) and does a vote.
def train_knn(X_train, y_train, X_val, y_val, n_neighbors: int = 25):
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)

    # Evaluate on validation spec features
    y_val_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"kNN (val): acc={acc:.3f}, macro-F1={macro_f1:.3f}")
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    return knn


# Train a multinomial Naive Bayes classifier on TEXT-ONLY features
# Inputs are bad-of-words counts from descriptions
# Model learns how word frequencies relate to each price band
def train_naive_bayes_text(X_train_text, y_train, X_val_text, y_val):
   
    nb = MultinomialNB()
    nb.fit(X_train_text, y_train)

    # Evaluate on validation text features
    y_val_pred = nb.predict(X_val_text)
    acc = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"Naive Bayes (text-only) (val): acc={acc:.3f}, macro-F1={macro_f1:.3f}")
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    #Return trained MultinomialNB model
    return nb