# Previsão de conversão no paywall

Case técnico de Machine Learning: prever a conversão em assinatura de um usuário quando o paywall é exibido, a partir de dados de sessões de navegação.

Toda a análise, a modelagem e as respostas às 8 perguntas do teste estão em [notebook.ipynb](notebook.ipynb). O enunciado original está em [Teste ML vendas paywall.pdf](Teste%20ML%20vendas%20paywall.pdf).

## Principais conclusões

- **Só 5% das sessões convertem.** O problema é fortemente desbalanceado, então a avaliação usa PR-AUC, precision e recall, e não acurácia.
- **`total_time_spent` é vazamento de dados.** Incluí-la leva o ROC-AUC de teste de 0,70 para ~1,00, porque o valor registrado é a duração total da sessão, incluindo o que acontece depois da conversão. Ela foi excluída do modelo.
- **O sinal está na sessão, não no usuário.** O histórico do usuário não prevê conversão futura: a taxa de conversão na primeira metade das sessões de cada usuário não se correlaciona com a da segunda metade (r = 0,02), e a variância entre usuários é a esperada por puro acaso.
- **Regressão Logística e CatBoost empatam** na previsão da sessão atual (ROC-AUC 0,70 e 0,69; PR-AUC 0,134 e 0,120, contra um baseline de 0,050). A logística é a escolhida, por simplicidade e interpretabilidade.
- **Recomendação de negócio:** ser mais agressivo com o paywall em sessões de alto engajamento. A conversão sobe de 1,6% em sessões com 0–1 artigos para 14,8% com 6 ou mais. Dispositivo e fim de semana não discriminam.

## Dados

[data.csv](data.csv) tem 10.000 sessões de 299 usuários, uma sessão por linha.

| Coluna | Descrição |
|---|---|
| `session_id` | identificador sequencial da sessão, usado como ordem temporal |
| `user_id` | identificador do usuário |
| `articles_viewed` | artigos lidos na sessão |
| `avg_scroll_depth` | profundidade média de rolagem (%) |
| `total_time_spent` | duração da sessão (s) |
| `is_weekend` | sessão em fim de semana (0/1) |
| `device` | desktop, mobile ou tablet |
| `converted` | a sessão terminou em conversão (0/1), o alvo |

O dataset não tem data/hora. A análise assume que `session_id` crescente reflete a ordem cronológica.

## Abordagem

1. **Exploração** guiada por perguntas: desbalanceamento, conversões repetidas por usuário e relação entre engajamento e conversão.
2. **Features de histórico do usuário** (sessões, tempo mediano, artigos e conversões anteriores), calculadas só com sessões estritamente anteriores à atual.
3. **Duas tarefas de previsão**, com split cronológico 80/20:
   - conversão na sessão atual (`converted`);
   - conversão nas próximas 5 sessões (`will_convert_next_k`), com descarte das linhas sem futuro observado para evitar viés de censura.
4. **Comparação** entre Regressão Logística e CatBoost, ambos com pesos de classe balanceados.
5. **Teste de leakage** de `total_time_spent` e verificação do teto de sinal do histórico do usuário.
6. **Respostas às 8 perguntas do teste**, cada uma com a evidência correspondente (última seção do notebook).

## Resultados

Métricas no conjunto de teste (20% mais recente):

| Tarefa | Modelo | ROC-AUC | PR-AUC | Baseline PR-AUC |
|---|---|---|---|---|
| Sessão atual | Regressão Logística | 0,699 | 0,134 | 0,050 |
| Sessão atual | CatBoost | 0,693 | 0,120 | 0,050 |
| Próximas 5 sessões | Regressão Logística | 0,429 | 0,205 | 0,235 |
| Próximas 5 sessões | CatBoost | 0,555 | 0,287 | 0,235 |

O modelo da sessão atual tem sinal real. O das próximas sessões não sai do trivial, por ausência de sinal nos dados.

## Como executar

Requer Python 3 e as bibliotecas abaixo:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn catboost jupyter
```

Abra [notebook.ipynb](notebook.ipynb) na pasta do projeto e execute as células em ordem, do início ao fim. As células de resposta dependem das variáveis criadas nas seções anteriores.

O treino do CatBoost gera a pasta `catboost_info/`, que contém apenas logs e pode ser apagada.

## Estrutura

```
├── notebook.ipynb                  análise, modelagem e respostas
├── data.csv                        dados das sessões
├── Teste ML vendas paywall.pdf     enunciado do teste
└── catboost_info/                  logs de treino do CatBoost (gerados)
```
