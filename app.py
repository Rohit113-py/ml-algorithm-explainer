import streamlit as st
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import plotly.express as px
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="ML Algorithm Explainer", layout="wide")

# -------------------------
# OPENAI API KEY
# -------------------------
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

with st.sidebar:
    st.title("🔑 API Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key
    )
    if api_key:
        st.session_state.openai_api_key = api_key

# -------------------------
# ML EXPLAINER CLASS
# -------------------------
class MLExplainer:
    def __init__(self):
        self.algorithms = [
            "Linear Regression",
            "Logistic Regression",
            "Decision Trees",
            "Random Forest",
            "Support Vector Machines",
            "K-Means Clustering",
            "Neural Networks",
            "Gradient Boosting"
        ]

    def get_ai_explanation(self, algorithm, level="beginner"):
        """Get AI-powered explanation using OpenAI"""
        if not st.session_state.openai_api_key:
            return f"""
### {algorithm}
**No API key provided**, so here's a built-in explanation:

- **What it does:** {self.get_fallback_explanation(algorithm)["what"]}
- **How it works:** {self.get_fallback_explanation(algorithm)["how"]}
- **When to use it:** {self.get_fallback_explanation(algorithm)["when"]}
- **Pitfalls:** {self.get_fallback_explanation(algorithm)["pitfalls"]}
"""

        prompt = f"""
Explain the {algorithm} machine learning algorithm for a {level} audience.

Include:
1. What it does
2. How it works (simple analogy)
3. Key parameters
4. When to use it
5. Common pitfalls

Keep it simple, engaging, and use bullet points.
"""

        try:
            client = OpenAI(api_key=st.session_state.openai_api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"""
### AI Error
Could not fetch explanation from OpenAI.

**Error:** `{e}`

### Fallback Explanation
- **What it does:** {self.get_fallback_explanation(algorithm)["what"]}
- **How it works:** {self.get_fallback_explanation(algorithm)["how"]}
- **When to use it:** {self.get_fallback_explanation(algorithm)["when"]}
- **Pitfalls:** {self.get_fallback_explanation(algorithm)["pitfalls"]}
"""

    def get_fallback_explanation(self, algorithm):
        explanations = {
            "Linear Regression": {
                "what": "Predicts a continuous value using a straight-line relationship.",
                "how": "Fits the best possible line through data points.",
                "when": "Use when predicting numbers like price, salary, or temperature.",
                "pitfalls": "Poor for non-linear data and sensitive to outliers."
            },
            "Logistic Regression": {
                "what": "Used for classification problems.",
                "how": "Calculates probability and assigns classes.",
                "when": "Use for binary or simple classification tasks.",
                "pitfalls": "Does not handle complex non-linear patterns well."
            },
            "Decision Trees": {
                "what": "Splits data into branches based on rules.",
                "how": "Asks a sequence of yes/no questions.",
                "when": "Use when interpretability is important.",
                "pitfalls": "Can overfit easily if too deep."
            },
            "Random Forest": {
                "what": "Combines many decision trees for better accuracy.",
                "how": "Each tree votes for the final answer.",
                "when": "Use for strong general-purpose classification.",
                "pitfalls": "Less interpretable than a single tree."
            },
            "Support Vector Machines": {
                "what": "Finds the best boundary between classes.",
                "how": "Draws the widest possible margin between groups.",
                "when": "Use for smaller, high-dimensional datasets.",
                "pitfalls": "Can be slow on large datasets."
            },
            "K-Means Clustering": {
                "what": "Groups similar data points into clusters.",
                "how": "Finds cluster centers and assigns points to the nearest center.",
                "when": "Use for unsupervised grouping.",
                "pitfalls": "Needs number of clusters заранее and struggles with irregular shapes."
            },
            "Neural Networks": {
                "what": "Learns complex patterns using layers of neurons.",
                "how": "Passes information through connected layers.",
                "when": "Use for complex tasks like image, text, and speech.",
                "pitfalls": "Needs more data and tuning."
            },
            "Gradient Boosting": {
                "what": "Builds models step by step to fix previous mistakes.",
                "how": "Each new model improves errors of the last one.",
                "when": "Use for high-performance structured/tabular data tasks.",
                "pitfalls": "Can overfit and be slower to train."
            }
        }
        return explanations.get(algorithm, {
            "what": "A machine learning algorithm.",
            "how": "Learns patterns from data.",
            "when": "Use depending on your task.",
            "pitfalls": "Requires proper tuning."
        })

    def get_model_code(self, algorithm):
        model_map = {
            "Linear Regression": "LinearRegression()",
            "Logistic Regression": "LogisticRegression(random_state=42)",
            "Decision Trees": "DecisionTreeClassifier(max_depth=3, random_state=42)",
            "Random Forest": "RandomForestClassifier(n_estimators=100, random_state=42)",
            "Support Vector Machines": "SVC(kernel='linear', random_state=42)",
            "K-Means Clustering": "KMeans(n_clusters=3, random_state=42)",
            "Neural Networks": "MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)",
            "Gradient Boosting": "GradientBoostingClassifier(random_state=42)"
        }
        return model_map.get(algorithm, "LogisticRegression()")

    def demo_algorithm(self, algorithm_name):
        iris = datasets.load_iris()
        X, y = iris.data[:, :2], iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        if algorithm_name == "Linear Regression":
            # Use synthetic regression target
            y_reg = iris.data[:, 2]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_reg, test_size=0.3, random_state=42
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            return model, X_test, y_test, y_pred, score, "mse"

        elif algorithm_name == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif algorithm_name == "Decision Trees":
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        elif algorithm_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif algorithm_name == "Support Vector Machines":
            model = SVC(kernel='linear', random_state=42)
        elif algorithm_name == "Neural Networks":
            model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        elif algorithm_name == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif algorithm_name == "K-Means Clustering":
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(X)
            y_pred = model.labels_
            return model, X, y, y_pred, None, "cluster"
        else:
            model = LogisticRegression(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, X_test, y_test, y_pred, accuracy, "classification"


# -------------------------
# INIT
# -------------------------
explainer = MLExplainer()

# -------------------------
# MAIN UI
# -------------------------
st.title("🤖 AI-Powered Machine Learning Algorithm Explainer")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    selected_algo = st.selectbox(
        "Choose an algorithm to explore:",
        explainer.algorithms
    )

with col2:
    explanation_level = st.selectbox(
        "Explanation Level:",
        ["beginner", "intermediate", "advanced"]
    )
    run_demo = st.checkbox("Run Live Demo", value=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📖 AI Explanation", "📊 Visualization", "💻 Code Example", "⚡ Live Demo"]
)

# -------------------------
# TAB 1: EXPLANATION
# -------------------------
with tab1:
    st.subheader(f"🤖 AI Explanation: {selected_algo}")
    ai_explanation = explainer.get_ai_explanation(selected_algo, explanation_level)
    st.markdown(ai_explanation)

    st.subheader("📋 Quick Facts")
    col_a, col_b, col_c = st.columns(3)

    task_type_map = {
        "Linear Regression": "Regression",
        "Logistic Regression": "Classification",
        "Decision Trees": "Classification",
        "Random Forest": "Classification",
        "Support Vector Machines": "Classification",
        "K-Means Clustering": "Clustering",
        "Neural Networks": "Classification",
        "Gradient Boosting": "Classification"
    }

    with col_a:
        st.metric("Task Type", task_type_map[selected_algo])
    with col_b:
        st.metric("Complexity", "Medium")
    with col_c:
        st.metric("Interpretability", "High" if selected_algo in ["Decision Trees", "Linear Regression", "Logistic Regression"] else "Medium")

# -------------------------
# TAB 2: VISUALIZATION
# -------------------------
with tab2:
    st.subheader(f"📊 Visualizing {selected_algo}")

    if selected_algo == "Decision Trees":
        fig, ax = plt.subplots(figsize=(14, 8))
        iris = datasets.load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        plot_tree(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True,
            ax=ax
        )
        st.pyplot(fig)

    elif selected_algo == "Support Vector Machines":
        X, y = datasets.make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
        model = SVC(kernel='rbf', gamma=10)
        model.fit(X, y)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)

        xx, yy = np.meshgrid(
            np.linspace(-1.5, 1.5, 200),
            np.linspace(-1.5, 1.5, 200)
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        ax.set_title("SVM Decision Boundary")

        st.pyplot(fig)

    elif selected_algo == "K-Means Clustering":
        X, _ = datasets.make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=300, marker='X', c='red')
        ax.set_title("K-Means Clustering")
        st.pyplot(fig)

    else:
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target.astype(str)

        fig = px.scatter_3d(
            df,
            x=iris.feature_names[0],
            y=iris.feature_names[1],
            z=iris.feature_names[2],
            color="target",
            title=f"{selected_algo} - Sample Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 3: CODE EXAMPLE
# -------------------------
with tab3:
    st.subheader("💻 Ready-to-Run Code Example")

    model_code = explainer.get_model_code(selected_algo)

    code_example = f'''# {selected_algo} - Complete Working Example
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = {model_code}

# Train & Predict
if "{selected_algo}" == "K-Means Clustering":
    model.fit(X)
    predictions = model.labels_
    print("Cluster labels:", predictions[:10])
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
'''

    st.code(code_example, language="python")

# -------------------------
# TAB 4: LIVE DEMO
# -------------------------
with tab4:
    st.subheader("⚡ Live Demo")

    if run_demo:
        model, X_data, y_true, y_pred, score, task_type = explainer.demo_algorithm(selected_algo)

        if task_type == "classification":
            st.success(f"✅ Accuracy: **{score:.3f}**")

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            st.pyplot(fig)

            demo_df = pd.DataFrame({
                "Feature 1": X_data[:, 0],
                "Feature 2": X_data[:, 1],
                "Actual": y_true,
                "Predicted": y_pred
            })
            st.dataframe(demo_df.head(10), use_container_width=True)

        elif task_type == "mse":
            st.info(f"📉 Mean Squared Error: **{score:.3f}**")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_true, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        elif task_type == "cluster":
            st.info("📌 K-Means clustering completed successfully.")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_data[:, 0], X_data[:, 1], c=y_pred, cmap='viridis', s=50)
            ax.set_title("Cluster Assignments")
            st.pyplot(fig)

    else:
        st.warning("Enable **Run Live Demo** checkbox to see results.")








