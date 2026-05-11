import pickle
import spacy
import streamlit as st


@st.cache_resource

def load_model():
    with open("model.pkl", "rb") as  f:
        return pickle.load(f)
    

@st.cache_resource
def load_npl():
    return spacy.blank("pt")

pipeline = load_model()
nlp = load_npl()

def preprocess (text : str) -> str:
    """Tokeniza , minuscula, e remove pontuação / espaços extras."""
    doc= nlp(str(text).lower())
    tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
    return " ".join(tokens)

#interface

st.title("Classificador de Produtos")
st.write ("Digite um produto para prever sua categoria.")

produto = st.text_input ("Nome do Produto", placeholder = "Ex: Smartphone Sansung Galaxy A54 128GB")


if produto :
    texto = preprocess(produto)
    categoria = pipeline.predict([texto])[0]
    proba = pipeline.predict_proba([texto])[0]
    classes = pipeline.classes_

    st.success (f"**Categoria Prevista:** {categoria}")

    st.subheader("Probabilidade por categoria")
    proba_dict = dict(sorted(zip(classes, proba), key = lambda x: x[1], reverse= True))
    for cat, prob in proba_dict.items():
        st.progress(float(prob), text=f"{cat}: {prob*100:.1f}%")