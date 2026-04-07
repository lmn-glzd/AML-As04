import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                               GradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, log_loss,
                              precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, PolynomialFeatures,
                                   RobustScaler, SplineTransformer)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Project 4 – Bank Marketing",
    page_icon="🏦",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="Set2")
RANDOM_STATE = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            'janiobachmann/bank-marketing-dataset',
            'bank.csv',
        )
    except Exception:
        url = ("https://raw.githubusercontent.com/dsrscientist/"
               "dataset1/master/bank.csv")
        df = pd.read_csv(url, sep=";")
    return df


@st.cache_data(show_spinner="Preprocessing…")
def preprocess(_df):
    df = _df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    df['deposit'] = (
        df['deposit'].astype(str).str.strip().str.lower()
        .map({'yes': 1, 'no': 0})
    )
    df = df.dropna(subset=['deposit']).reset_index(drop=True)
    df['deposit'] = df['deposit'].astype(int)

    X = df.drop(columns=['deposit']).copy()
    y = df['deposit'].copy()

    if 'duration' in X.columns:
        X = X.drop(columns=['duration'])
    if 'pdays' in X.columns:
        X['was_previously_contacted'] = (X['pdays'] != -1).astype(int)
        X['pdays_clean'] = X['pdays'].replace(-1, 0)
        X = X.drop(columns=['pdays'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    numeric_features  = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    return X, y, X_train, X_test, y_train, y_test, numeric_features, categorical_features


def make_scaled_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', RobustScaler()),
        ]), numeric_features),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_features),
    ])


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, 'predict_proba')
               else model.decision_function(X_test)
               if hasattr(model, 'decision_function') else None)
    return {
        'model': name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_score), 4) if y_score is not None else np.nan,
        '_model_obj': model,
        '_y_pred': y_pred,
        '_y_score': y_score,
    }


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf


# ── App ───────────────────────────────────────────────────────────────────────
st.title("🏦 AML Project 4 — Bank Marketing Classification")
st.markdown(
    "**Non-Linearity · Tree-Based Methods · SVM** | Bank Marketing Dataset"
)

# Load data
raw_df = load_data()
X, y, X_train, X_test, y_train, y_test, num_feats, cat_feats = preprocess(raw_df)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 EDA", "🔬 Non-Linear & SVM", "🌲 Tree Methods", "🏆 Final Comparison"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    col_l, col_r = st.columns(2)
    with col_l:
        st.metric("Rows", f"{raw_df.shape[0]:,}")
        st.metric("Features", raw_df.shape[1] - 1)
    with col_r:
        vc = raw_df['deposit'].value_counts()
        st.metric("Positive rate (yes)", f"{(vc.get('yes',0)/len(raw_df)*100):.1f}%")
        st.metric("Negative rate (no)",  f"{(vc.get('no',0)/len(raw_df)*100):.1f}%")

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

    # Target distribution
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=raw_df, x='deposit', ax=ax, palette='Set2')
    ax.set_title("Deposit (Target)")
    st.image(fig_to_img(fig))

    # Numeric histograms
    st.subheader("Numeric Feature Distributions")
    numeric_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(raw_df[col].dropna(), bins=20, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.image(fig_to_img(fig))

    # Correlation matrix
    st.subheader("Correlation Matrix")
    df_enc = raw_df.copy()
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df_enc.corr().round(2), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, linewidths=0.5,
                annot_kws={'size': 6})
    ax.set_title("Correlation Matrix — All Features")
    plt.tight_layout()
    st.image(fig_to_img(fig))

    # Box plots
    st.subheader("Box Plots (Outliers)")
    fig, axes = plt.subplots(
        (n + 2) // 3, 3, figsize=(14, ((n + 2) // 3) * 3)
    )
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=raw_df[col], ax=axes[i])
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.image(fig_to_img(fig))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Non-Linear & SVM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Non-Linear Models & SVM")

    top_numeric = ['pdays_clean', 'previous', 'balance']
    top_numeric = [f for f in top_numeric if f in num_feats]

    def nl_preprocessor(selected, transformer):
        other = [c for c in num_feats if c not in selected]
        return ColumnTransformer([
            ('sel', transformer, selected),
            ('oth', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('scl', RobustScaler())]), other),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]),
             cat_feats),
        ], remainder='drop')

    gam_steps = [
        (f'spline_{f}', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', RobustScaler()),
            ('spl', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ]), [f]) for f in top_numeric
    ]
    gam_preprocessor = ColumnTransformer(
        gam_steps + [
            ('oth', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('scl', RobustScaler())]),
             [c for c in num_feats if c not in top_numeric]),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]),
             cat_feats),
        ], remainder='drop'
    )

    ALL_MODELS = {
        'Baseline Logistic Regression': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        'Polynomial Logistic': Pipeline([
            ('pre', nl_preprocessor(top_numeric, Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scl', RobustScaler()),
                ('pol', PolynomialFeatures(degree=2, include_bias=False)),
            ]))),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        'Step Function Logistic': Pipeline([
            ('pre', nl_preprocessor(top_numeric, Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('bin', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
            ]))),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        'Spline Logistic': Pipeline([
            ('pre', nl_preprocessor(top_numeric, Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scl', RobustScaler()),
                ('spl', SplineTransformer(n_knots=6, degree=3, include_bias=False)),
            ]))),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        'Local Classification (KNN)': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', KNeighborsClassifier(n_neighbors=25)),
        ]),
        'GAM-style Logistic': Pipeline([
            ('pre', gam_preprocessor),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        'SVM Linear': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', SVC(kernel='linear', probability=True)),
        ]),
        'SVM Polynomial': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', SVC(kernel='poly', degree=3, probability=True)),
        ]),
        'SVM RBF': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', SVC(kernel='rbf', probability=True)),
        ]),
        'SVM Sigmoid': Pipeline([
            ('pre', make_scaled_preprocessor(num_feats, cat_feats)),
            ('clf', SVC(kernel='sigmoid', probability=True)),
        ]),
    }

    selected_models = st.multiselect(
        "Select models to train",
        list(ALL_MODELS.keys()),
        default=['Baseline Logistic Regression', 'Spline Logistic', 'SVM RBF'],
    )

    if st.button("▶ Train Selected Models", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            results = []
            prog = st.progress(0)
            for i, name in enumerate(selected_models):
                with st.spinner(f"Training {name}…"):
                    res = evaluate(name, ALL_MODELS[name],
                                   X_train, X_test, y_train, y_test)
                    results.append(res)
                prog.progress((i + 1) / len(selected_models))

            st.session_state['nl_svm_results'] = results
            st.success("Done!")

    if 'nl_svm_results' in st.session_state:
        results = st.session_state['nl_svm_results']
        metric_cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        df_res = pd.DataFrame([{k: v for k, v in r.items() if k in metric_cols}
                                for r in results])
        df_res = df_res.sort_values('roc_auc', ascending=False).reset_index(drop=True)

        st.subheader("Results")
        st.dataframe(df_res.style.background_gradient(
            subset=['accuracy', 'f1', 'roc_auc'], cmap='Blues'), use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(9, max(3, len(df_res) * 0.5)))
        colors = sns.color_palette("Set2", len(df_res))
        bars = ax.barh(df_res['model'][::-1], df_res['roc_auc'][::-1], color=colors)
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Model Comparison — ROC-AUC')
        ax.set_xlim(0.5, 1.0)
        for bar, val in zip(bars, df_res['roc_auc'][::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)
        plt.tight_layout()
        st.image(fig_to_img(fig))

        # ROC curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in results:
            if r['_y_score'] is not None:
                fpr, tpr, _ = roc_curve(y_test, r['_y_score'])
                ax.plot(fpr, tpr, lw=2, label=f"{r['model']} ({r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        st.image(fig_to_img(fig))

        # Confusion matrices
        st.subheader("Confusion Matrices")
        n_models = len(results)
        cols = min(3, n_models)
        rows_ = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 4 * rows_))
        if n_models == 1:
            axes = np.array([[axes]])
        elif rows_ == 1:
            axes = axes.reshape(1, -1)
        for idx, r in enumerate(results):
            ax = axes[idx // cols][idx % cols]
            cm = confusion_matrix(y_test, r['_y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                        linewidths=0.5)
            ax.set_title(r['model'], fontsize=8)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        for j in range(idx + 1, rows_ * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.tight_layout()
        st.image(fig_to_img(fig))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tree Methods
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Tree-Based Methods")

    # Encode for trees
    @st.cache_data
    def encode_for_trees(_df):
        df_enc = _df.copy()
        if 'duration' in df_enc.columns:
            df_enc = df_enc.drop(columns=['duration'])
        le = LabelEncoder()
        for col in df_enc.select_dtypes(include='object').columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        X_t = df_enc.drop('deposit', axis=1)
        y_t = df_enc['deposit']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_t, y_t, test_size=0.25, random_state=RANDOM_STATE, stratify=y_t)
        return X_t, y_t, X_tr, X_te, y_tr, y_te

    X_tree, y_tree, X_tr, X_te, y_tr, y_te = encode_for_trees(raw_df)

    st.subheader("Configure Tree Models")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Bagging**")
        bag_n = st.slider("n_estimators", 50, 300, 200, 50, key='bag_n')

    with c2:
        st.markdown("**Random Forest**")
        rf_n  = st.slider("n_estimators", 50, 500, 300, 50, key='rf_n')
        rf_depth = st.selectbox("max_depth", [None, 5, 10, 15, 20], key='rf_d')

    with c3:
        st.markdown("**Gradient Boosting**")
        gb_n = st.slider("n_estimators", 50, 500, 300, 50, key='gb_n')
        gb_lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], value=0.05, key='gb_lr')

    run_trees = st.multiselect(
        "Select tree methods to run",
        ['Bagging', 'Random Forest', 'Gradient Boosting'],
        default=['Bagging', 'Random Forest', 'Gradient Boosting'],
    )

    if st.button("▶ Train Tree Models", type="primary"):
        if not run_trees:
            st.warning("Select at least one method.")
        else:
            tree_results = {}
            prog2 = st.progress(0)

            if 'Bagging' in run_trees:
                with st.spinner("Training Bagging…"):
                    bag = BaggingClassifier(
                        estimator=DecisionTreeClassifier(max_features='sqrt'),
                        n_estimators=bag_n, max_samples=0.8,
                        bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
                    bag.fit(X_tr, y_tr)
                    tree_results['Bagging'] = {
                        'model': bag,
                        'pred': bag.predict(X_te),
                        'prob': bag.predict_proba(X_te)[:, 1],
                    }
                prog2.progress(1 / len(run_trees))

            if 'Random Forest' in run_trees:
                with st.spinner("Training Random Forest…"):
                    rf = RandomForestClassifier(
                        n_estimators=rf_n, max_features='sqrt',
                        max_depth=rf_depth, min_samples_leaf=2,
                        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
                    rf.fit(X_tr, y_tr)
                    tree_results['Random Forest'] = {
                        'model': rf,
                        'pred': rf.predict(X_te),
                        'prob': rf.predict_proba(X_te)[:, 1],
                        'feat_imp': pd.Series(rf.feature_importances_, index=X_tree.columns),
                    }
                prog2.progress(min(2 / len(run_trees), 1.0))

            if 'Gradient Boosting' in run_trees:
                with st.spinner("Training Gradient Boosting…"):
                    gb = GradientBoostingClassifier(
                        n_estimators=gb_n, learning_rate=gb_lr,
                        max_depth=4, subsample=0.8, min_samples_leaf=5,
                        random_state=RANDOM_STATE)
                    gb.fit(X_tr, y_tr)
                    tree_results['Gradient Boosting'] = {
                        'model': gb,
                        'pred': gb.predict(X_te),
                        'prob': gb.predict_proba(X_te)[:, 1],
                        'feat_imp': pd.Series(gb.feature_importances_, index=X_tree.columns),
                        'staged_loss': [log_loss(y_te, p[:, 1])
                                        for p in gb.staged_predict_proba(X_te)],
                        'train_score': gb.train_score_,
                    }
                prog2.progress(1.0)

            st.session_state['tree_results'] = tree_results
            st.success("Done!")

    if 'tree_results' in st.session_state:
        tree_results = st.session_state['tree_results']

        # Summary table
        rows = []
        for name, r in tree_results.items():
            rows.append({
                'Model': name,
                'Accuracy':  round(accuracy_score(y_te, r['pred']), 4),
                'Precision': round(precision_score(y_te, r['pred'], zero_division=0), 4),
                'Recall':    round(recall_score(y_te, r['pred'], zero_division=0), 4),
                'F1':        round(f1_score(y_te, r['pred'], zero_division=0), 4),
                'ROC-AUC':   round(roc_auc_score(y_te, r['prob']), 4),
            })
        df_tree_res = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False)
        st.subheader("Summary")
        st.dataframe(df_tree_res.style.background_gradient(
            subset=['Accuracy', 'F1', 'ROC-AUC'], cmap='Greens'), use_container_width=True)

        # Individual diagnostics
        for name, r in tree_results.items():
            st.subheader(f"📌 {name}")

            if name == 'Gradient Boosting' and 'staged_loss' in r:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'{name} — Diagnostic Report', fontsize=13, fontweight='bold')

                sl = r['staged_loss']
                ts = r['train_score']
                best_iter = int(np.argmin(sl))
                axes[0, 0].plot(ts, color='#2563EB', lw=2, label='Train Deviance')
                axes[0, 0].plot(sl, color='#DC2626', lw=2, label='Test Deviance')
                axes[0, 0].axvline(best_iter, color='#10B981', ls='--',
                                   label=f'Best iter = {best_iter}')
                axes[0, 0].set_title('Deviance vs. Iterations')
                axes[0, 0].legend()

                fi = r['feat_imp'].sort_values(ascending=False).head(10)
                colors_g = plt.cm.Greens_r(np.linspace(0.2, 0.85, len(fi)))
                axes[0, 1].barh(fi.index[::-1], fi.values[::-1],
                                color=colors_g[::-1], edgecolor='white')
                axes[0, 1].set_title('Top-10 Feature Importances')

                cm_ = confusion_matrix(y_te, r['pred'])
                sns.heatmap(cm_, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                axes[1, 0].set_title('Confusion Matrix')

                fpr_, tpr_, _ = roc_curve(y_te, r['prob'])
                auc_ = roc_auc_score(y_te, r['prob'])
                axes[1, 1].plot(fpr_, tpr_, color='#16A34A', lw=2, label=f'AUC={auc_:.4f}')
                axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1)
                axes[1, 1].set_title('ROC Curve')
                axes[1, 1].legend()
                plt.tight_layout()
                st.image(fig_to_img(fig))

            else:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fig.suptitle(f'{name}', fontsize=12, fontweight='bold')

                # Feature importance if available
                if 'feat_imp' in r:
                    fi = r['feat_imp'].sort_values(ascending=False).head(10)
                    colors_b = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(fi)))
                    axes[0].barh(fi.index[::-1], fi.values[::-1],
                                 color=colors_b[::-1], edgecolor='white')
                    axes[0].set_title('Top-10 Feature Importances')
                else:
                    axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center')
                    axes[0].set_title('Feature Importances')

                cm_ = confusion_matrix(y_te, r['pred'])
                sns.heatmap(cm_, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                axes[1].set_title('Confusion Matrix')

                fpr_, tpr_, _ = roc_curve(y_te, r['prob'])
                auc_ = roc_auc_score(y_te, r['prob'])
                axes[2].plot(fpr_, tpr_, color='#2563EB', lw=2, label=f'AUC={auc_:.4f}')
                axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
                axes[2].set_title('ROC Curve')
                axes[2].legend()
                plt.tight_layout()
                st.image(fig_to_img(fig))

            st.text(classification_report(y_te, r['pred'], target_names=['No', 'Yes']))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Final Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🏆 Final Model Comparison")

    nl_df, tree_df = pd.DataFrame(), pd.DataFrame()
    metric_cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    if 'nl_svm_results' in st.session_state:
        nl_df = pd.DataFrame([
            {k: v for k, v in r.items() if k in metric_cols}
            for r in st.session_state['nl_svm_results']
        ])

    if 'tree_results' in st.session_state:
        rows = []
        for name, r in st.session_state['tree_results'].items():
            rows.append({
                'model': name,
                'accuracy':  round(accuracy_score(y_te, r['pred']), 4),
                'precision': round(precision_score(y_te, r['pred'], zero_division=0), 4),
                'recall':    round(recall_score(y_te, r['pred'], zero_division=0), 4),
                'f1':        round(f1_score(y_te, r['pred'], zero_division=0), 4),
                'roc_auc':   round(roc_auc_score(y_te, r['prob']), 4),
            })
        tree_df = pd.DataFrame(rows)

    if nl_df.empty and tree_df.empty:
        st.info("Train models in the **Non-Linear & SVM** and **Tree Methods** tabs first.")
    else:
        final_df = pd.concat([nl_df, tree_df], ignore_index=True)
        final_df = (final_df.drop_duplicates('model')
                             .sort_values('roc_auc', ascending=False)
                             .reset_index(drop=True))

        st.dataframe(
            final_df.style.background_gradient(
                subset=['accuracy', 'f1', 'roc_auc'], cmap='RdYlGn'),
            use_container_width=True,
        )

        # Champion
        best = final_df.iloc[0]
        st.success(f"🥇 **Best model:** {best['model']} | "
                   f"ROC-AUC: {best['roc_auc']:.4f} | "
                   f"F1: {best['f1']:.4f} | "
                   f"Accuracy: {best['accuracy']:.4f}")

        # Multi-metric bar chart
        st.subheader("All Metrics Side-by-Side")
        melted = final_df.melt(id_vars='model',
                               value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                               var_name='metric', value_name='score')
        fig, ax = plt.subplots(figsize=(12, max(5, len(final_df) * 0.6)))
        sns.barplot(data=melted, x='score', y='model', hue='metric', ax=ax)
        ax.set_xlim(0.5, 1.0)
        ax.set_title('All Models — All Metrics')
        ax.set_xlabel('Score')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        st.image(fig_to_img(fig))

        # Download
        csv = final_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Results CSV", csv,
                           "final_comparison.csv", "text/csv")