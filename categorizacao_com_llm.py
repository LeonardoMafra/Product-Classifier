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


nome = "Banheira inflavel"
resposta = classificar_produto(nome)


print(f'A categoria do produto {nome} é : {resposta}')



