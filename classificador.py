import pandas as pd
import pickle
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#1.Carregar dados
df = pd.read_csv('base_tratada.csv', usecols=['product_name','site_category_lv1'])
df = df.dropna()
print(f'Total de registros: {len(df)}')
print(f'Categorias: {df["site_category_lv1"].nunique()}')


#processamento com spaCy
npl = spacy .blank("pt") # tokenizador em branco ( sem modelo de linguagem)


def preprocess (text : str) -> str:
    """Tokeniza , minuscula, e remove pontuação / espaços extras."""
    doc= npl(str(text).lower())
    tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
    return " ".join(tokens)


print("Pré-processamento texto...")

df["texto"] = df["product_name"].apply(preprocess)

X_train,X_test,y_train,y_test = train_test_split(
    df["texto"] , df["site_category_lv1"],
    test_size=0.2,random_state=42, stratify=df["site_category_lv1"])

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

#pipline: tfdf + regressão logistica

Pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features= 50_000)),
        ("clf", LogisticRegression(max_iter=1000, C=5, solver="lbfgs")),
    ]

)

print("Treinando modelo")

Pipeline.fit(X_train, y_train)

# AVALIAÇÃO

y_pred = Pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {acc:.4f} ({acc*100:.1f}%)")
print("\nRelatório completo:")
print(classification_report(y_test, y_pred))

#salvando modelo no pickle

MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(Pipeline, f)
    print(f"\nModelo salbo em '{MODEL_PATH}'")



# analise dos erros

erros = X_test.copy().to_frame()
erros["categoria_real"] = y_test.values
erros["categoria_prevista"] = y_pred

erros = erros[erros["categoria_real"] != erros["categoria_prevista"]]
erros["produto_original"] = df.loc[erros.index, "product_name"].values

erros = erros[["produto_original", "categoria_real", "categoria_prevista"]]
erros.to_csv("erros_modelo.csv", index=False)

print(f"\nTotal de erros: {len(erros)}")
print(erros.to_string(index=False))
