import os
import time
import pandas as pd
from groq                    import Groq
from dotenv                  import load_dotenv
from sklearn.metrics         import classification_report,accuracy_score
from sklearn.model_selection import train_test_split


load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SIMPLE_SIZE = 200
SLEEP_BETWEEN_CALLS = 0.3

CATEGORIAS = [
'Eletroportáteis',  
'Brinquedos',
'TV e Home Theater',  
'Celulares e Smartphones',
'Informática e Acessórios',    
'Utilidades Domésticas',
'Beleza e Perfumaria',            
'Informática',
'Eletrodomésticos',                    
'Bebês',
]

SYSTEM_PROMPT = f"""Você é um classificador de produtos de e-commerce brasileiro.
Sua tarefa é classificar o nome de um produto em exatamente UMA das categorias abaixo.

Categorias disponíveis:
{chr(10).join(f'- {c}' for c in CATEGORIAS)}

Regras:

1. Responda APENAS com o nome exato da categoia, sem explicações, sem pontuação extra.
2 não invente categorias novas.
3 Se houver duvida, escoha a categoria mais próxima da lista


Exemplos:
Produto: "Smartphone Samsung Galaxy A54 128GB"
Resposta: Celulares e Smartphones

Exemplos:
Produto: "Panela de Pressão Inox 4,5L Tramontina"
Resposta: Utilidades Domésticas

Exemplos:
Produto: "Notebook Dell Inspiron 15 Core 15"
Resposta: Informática

Exemplos:
Produto: "Smart TV LED 55" LG 4K"
Resposta: TV e Home Teacher

Exemplos:
Produto: "Fraldas Pampers Premium Care RN"
Resposta: Bebês

"""

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def classificar_produto(nome_produto : str) -> str:
    """Envia o nome do produto ao LLM e retorna a categoria prevista"""

    completion = client.chat.completions.create(
    model =MODEL,
    messages=[
    {
     "role": "system", "content": SYSTEM_PROMPT
    },

    {
    "role": "user", "content": f"produto; {nome_produto}"
    },
    ],
    temperature = 0,
    max_tokens = 30,
    )

    resposta = completion.choices[0].message.content.strip()

    # garante que a resposta esta na lista de categorias validas
    for cat in CATEGORIAS:
        if cat.lower() == resposta.lower():
            return cat
    
    #fallback : busca a categoria mais precisa: (evita erros)

    resposta_lower = resposta.lower()
    for cat in CATEGORIAS:
        if cat.lower() in resposta_lower or resposta_lower in cat.lower():
            return cat
     
     #caso nao encontrar nada , ira retornar como erro.
    return resposta


#carregar base de erros do modelo de ML

df = pd.read_csv("erros_modelo.csv")
print(f"{'-'*60}")
print(f"Base: erros_modelo.csv - {len(df)} casos")
print(f" Modelo LLM {MODEL}")
print(f"{'-'*60}")

#Classificar cada erro com o LLM

predicoes_llm = []

#  Classificar cada erro com o LLM
predicoes_llm = []
 
for i, row in enumerate(df.itertuples(), 1):
    pred = classificar_produto(row.produto_original)
    predicoes_llm.append(pred)
 
    status = "CERTO" if pred == row.categoria_real else "ERRADO"
    print(f"[{i:02d}/{len(df)}] {status}  '{row.produto_original}'")
    print(f"        Real: {row.categoria_real:<28} "
          f"ML errou: {row.categoria_prevista:<28} "
          f"LLM: {pred}")
 
    time.sleep(SLEEP_BETWEEN_CALLS)
 
# Métricas
y_true = df["categoria_real"].tolist()
y_pred = predicoes_llm
 
acc = accuracy_score(y_true, y_pred)
total        = len(df)
acertos_llm  = sum(p == r for p, r in zip(y_pred, y_true))
erros_llm    = total - acertos_llm
 
# Quantos erros do ML o LLM corrigiu
corrigidos = sum(
    p == r
    for p, r, ml in zip(y_pred, y_true, df["categoria_prevista"])
    if p != ml   # LLM deu resposta diferente do ML
)
 
print(f"\n{'='*60}")
print(f"  RESULTADO FINAL")
print(f"{'='*60}")
print(f"  Total de casos (erros do ML) : {total}")
print(f"  LLM acertou                  : {acertos_llm}  ({acc*100:.1f}%)")
print(f"  LLM também errou             : {erros_llm}  ({(1-acc)*100:.1f}%)")
print(f"{'='*60}")
 
print("\nRelatório por categoria:")
print(classification_report(y_true, y_pred, zero_division=0))
 
#  Salvar resultado detalhado
df_resultado = df.copy()
df_resultado["categoria_llm"]  = y_pred
df_resultado["llm_acertou"]    = df_resultado["categoria_llm"] == df_resultado["categoria_real"]
df_resultado["ml_vs_llm"] = df_resultado.apply(
    lambda r: "ambos erraram"   if not r["llm_acertou"] else
              "LLM corrigiu",
    axis=1,
)
 
df_resultado.to_excel("comparacao_ml_vs_llm.xlsx", index=False)
print("\nArquivo salvo: comparacao_ml_vs_llm.xlsx")
 
# Resumo final
print("\nResumo comparativo:")
print(df_resultado["ml_vs_llm"].value_counts().to_string())



