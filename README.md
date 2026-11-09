# CIFAR-10 | Classificação de imagens com TensorFlow/Keras

CNN para classificar imagens RGB de 32 × 32 nas dez classes do CIFAR-10. O projeto segue uma organização inspirada em Cookiecutter Data Science e Made With ML.

## Protocolo experimental

1. O carregador verifica o SHA-256 do arquivo oficial e lê os cinco lotes de desenvolvimento. O lote oficial de teste fica reservado para a avaliação final.
2. A divisão estratificada usa a semente 42 e reserva 20% do desenvolvimento para validação. O treino grava os índices em `data/processed/split_indices.npz`.
3. A CNN aplica reescala fixa de pixels para o intervalo [0, 1]. Espelhamento, translação e dropout atuam no treino. As estatísticas de Batch Normalization são aprendidas no treino.
4. O checkpoint é selecionado pela menor perda de validação. Acurácia, Precision, Recall e F1 são calculados em modo de inferência com o mesmo checkpoint para treino e validação.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `configs/` | Divisão, semente e hiperparâmetros da CNN |
| `src/cifar10_classifier/` | Carregamento, arquitetura, treino, métricas e gráficos |
| `scripts/` | Entradas para treinar e avaliar |
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Distribuição, exemplos visuais, pixels e duplicatas exatas |
| [02_treino_validacao.ipynb](notebooks/02_treino_validacao.ipynb) | Histórico por época, curvas e checkpoint selecionado |
| [03_metricas_validacao.ipynb](notebooks/03_metricas_validacao.ipynb) | Comparação dos conjuntos, métricas por classe, matriz e exemplos de erros |
| [04_comparacao_cnns.ipynb](notebooks/04_comparacao_cnns.ipynb) | Comparação da CNN leve com a residual |
| [05_metricas_teste.ipynb](notebooks/05_metricas_teste.ipynb) | Avaliação final no teste oficial |
| `reports/` e `models/` | Artefatos locais da execução |
| `tests/` e `.github/workflows/` | Testes unitários e CI |

## Resultados da rodada inicial

A EDA confirmou 4.000 imagens por classe no treino e 1.000 por classe na validação. A auditoria de pixels encontrou zero cópias exatas entre os dois conjuntos.

A época 7 apresentou a menor perda de validação (0,9330). Com esse checkpoint, a CNN obteve:

| Métrica | Treino | Validação |
| --- | ---: | ---: |
| Acurácia | 68,24% | 66,97% |
| Precision macro | 68,54% | 67,33% |
| Recall macro | 68,24% | 66,97% |
| F1 macro | 67,38% | 66,17% |

Gato foi a classe com menor F1 (38,51%) e recall de 30,50%. A maior confusão foi gato → cachorro (318 de 1.000 gatos). Para a próxima rodada, vale ampliar o número de épocas com a mesma configuração e comparar F1 macro, recall de gato e a evolução da perda de validação.

## Reproduzir a rodada inicial

A configuração `configs/baseline_cpu.yaml` usa três blocos convolucionais com 16, 32 e 64 filtros e orçamento de oito épocas na CPU. A execução registrada usou Python 3.10, TensorFlow 2.20.0, Keras 3.12.0, NumPy 2.2.5 e scikit-learn 1.7.1.

Execute os comandos na raiz do projeto com um ambiente virtual ativo.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train.py --config configs/baseline_cpu.yaml
python scripts/evaluate_validation.py --config configs/baseline_cpu.yaml
jupyter lab
```

O treino gera `models/best_model.keras`, `reports/history.csv`, `reports/run_metadata.json` e `reports/figures/training_curves.png`. A avaliação gera `reports/train_metrics.json`, `reports/validation_metrics.json`, o relatório por classe, a lista de erros e os gráficos de validação. Os notebooks 02 e 03 exibem os artefatos da última execução local quando são executados novamente.

Para verificar as funções de divisão e métricas e o isolamento do teste, execute `python -m pytest -q`.

## Comparação com a CNN residual

A residual tem quatro blocos com conexões de atalho e 622.442 parâmetros. A comparação reutilizou exatamente os mesmos índices, semente e orçamento de oito épocas. Os checkpoints foram escolhidos pela menor perda de validação de cada rodada.

| Métrica na validação | CNN leve | CNN residual |
| --- | ---: | ---: |
| Acurácia | 66,97% | 71,84% |
| Precision macro | 67,33% | 75,15% |
| Recall macro | 66,97% | 71,84% |
| F1 macro | 66,17% | 71,42% |

O F1 de gato subiu de 38,51% para 54,01%, enquanto pássaro e sapo perderam F1. A acurácia de validação da residual caiu na época 7 e se recuperou na época 8, que foi selecionada. O [notebook de comparação](notebooks/04_comparacao_cnns.ipynb) inclui as curvas, os pares de classes e os erros por imagem.

Para reproduzir a residual, execute na raiz:

```bash
python scripts/train.py --config configs/residual_cpu.yaml
python scripts/evaluate_validation.py --config configs/residual_cpu.yaml
```

O checkpoint fica em `models/residual_cpu/` e os relatórios em `reports/residual_cpu/`.

## Continuação da CNN residual até 60 épocas

A configuração `configs/residual_60_cpu.yaml` retoma o modelo e o otimizador do checkpoint da época 8. A divisão foi verificada pelos índices salvos. Os callbacks foram reiniciados na continuação e o orçamento foi ampliado para 60 épocas totais.

| Métrica na validação | Rodada de 8 épocas | Rodada de 60 épocas |
| --- | ---: | ---: |
| Acurácia | 71,84% | 85,52% |
| Taxa de erro | 28,16% | 14,48% |
| Precision macro | 75,15% | 85,55% |
| Recall macro | 71,84% | 85,52% |
| F1 macro | 71,42% | 85,47% |

O checkpoint selecionado foi o da época 34, com perda de validação de 0,6311. Na época 60, a perda de validação foi 0,6419. O [notebook comparativo](notebooks/04_comparacao_cnns.ipynb) inclui as 60 épocas, a evolução da taxa de aprendizado e as confusões por classe.

Depois de executar a rodada residual de oito épocas, rode os comandos abaixo.

```bash
python scripts/train.py --config configs/residual_60_cpu.yaml
python scripts/evaluate_validation.py --config configs/residual_60_cpu.yaml
```

O checkpoint fica em `models/residual_60_cpu/` e as métricas em `reports/residual_60_cpu/`.

## Experimento de brilho e contraste

Os ajustes finos `residual_luz` e `residual_luz_controle` partem do checkpoint da época 34 de `residual_60_cpu`. Cada braço treinou por seis épocas adicionais. A variante aplicou brilho de ±5% e contraste de ±10% apenas às imagens de treino.

| Métrica na validação | Partida | Controle | Brilho e contraste |
| --- | ---: | ---: | ---: |
| Acurácia | 85,52% | 85,44% | 85,62% |
| Precision macro | 85,55% | 85,38% | 85,57% |
| Recall macro | 85,52% | 85,44% | 85,62% |
| F1 macro | 85,47% | 85,37% | 85,56% |
| Perda | 0,6311 | 0,6162 | 0,6083 |

As curvas e as confusões estão no [notebook comparativo](notebooks/04_comparacao_cnns.ipynb). Para repetir o experimento após a rodada de 60 épocas, execute os comandos abaixo.

```bash
python scripts/train.py --config configs/residual_luz_controle.yaml
python scripts/evaluate_validation.py --config configs/residual_luz_controle.yaml
python scripts/train.py --config configs/residual_luz.yaml
python scripts/evaluate_validation.py --config configs/residual_luz.yaml
```

## Interpolação bicúbica

O ajuste fino `residual_bicubica` partiu do checkpoint da época 34 e ampliou as imagens de 32 × 32 para 64 × 64 dentro da CNN. O controle usou 32 × 32. As duas rodadas usaram os mesmos índices e seis épocas adicionais, com brilho e contraste desativados.

| Métrica na validação | Controle 32 × 32 | Bicúbica 64 × 64 |
| --- | ---: | ---: |
| Acurácia | 85,44% | 84,51% |
| Precision macro | 85,38% | 84,95% |
| Recall macro | 85,44% | 84,51% |
| F1 macro | 85,37% | 84,53% |
| Perda | 0,6162 | 0,5855 |

O [notebook comparativo](notebooks/04_comparacao_cnns.ipynb) reúne as curvas e os erros por classe. Para repetir a rodada após o treinamento de 60 épocas, execute os comandos abaixo.

```bash
python scripts/train.py --config configs/residual_bicubica.yaml
python scripts/evaluate_validation.py --config configs/residual_bicubica.yaml
```

## Avaliação final no teste oficial

O manifesto `configs/final_test.yaml` identifica pelo SHA-256 o checkpoint da época 34 da CNN residual treinada até 60 épocas. A avaliação usou as 10.000 imagens do lote oficial e registrou **85,27% de acurácia** e **85,21% de F1 macro**.

O [notebook de teste](notebooks/05_metricas_teste.ipynb) contém os resultados agregados e por classe, a matriz de confusão e as curvas ROC e Precision–Recall. Os dados usados no notebook ficam em `reports/final_test/`.

Para gerar o relatório final com o checkpoint congelado, use o comando abaixo. Se o relatório já estiver disponível, o comando devolve as métricas gravadas.

```bash
python scripts/evaluate_test.py
```

## Modelo serializado e API

O exportador gera `models/final/cifar10_residual.keras` para inferência e grava a origem e o SHA-256 em `models/final/manifest.json`. Ele usa o checkpoint da época 34 da rodada de 60 épocas, valida o SHA-256 do manifesto final e compara as previsões antes e depois da exportação.

Na raiz do projeto, com o ambiente virtual ativo, execute os comandos abaixo.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[api]"
python scripts/export_model.py
python -m uvicorn cifar10_classifier.api:app --host 127.0.0.1 --port 8000
```

Abra a [documentação interativa](http://127.0.0.1:8000/docs) para enviar uma imagem. Em outro terminal, também é possível usar os comandos abaixo.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -F "file=@C:\caminho\imagem.jpg" http://127.0.0.1:8000/predict
```

`POST /predict` aceita JPEG, PNG ou WebP estáticos, com até 5 MiB e 16 megapixels. A API corrige a orientação EXIF, converte para RGB e ajusta a imagem para 32 × 32 com recorte central e interpolação bilinear. A resposta inclui a classe prevista, a confiança e as probabilidades das dez classes do CIFAR-10. `GET /health` identifica a rodada e a época do modelo carregado.

**Fonte dos dados** [CIFAR-10 oficial](https://www.cs.toronto.edu/~kriz/cifar.html).
