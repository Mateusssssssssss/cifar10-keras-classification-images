# AGENTS.md — Diretrizes de desenvolvimento

Seguir organização Cookiecutter Data Science / Made With ML. Código reutilizável em `src/`; notebooks para EDA, prototipação e análise visual. Garantir rastreabilidade e reprodutibilidade, testar funções puras e manter documentação clara.

**Regra inegociável:** usar CIFAR-10 oficial de desenvolvimento para train/validation e manter teste oficial isolado até a fase final. Nunca ajustar arquitetura ou hiperparâmetros com base no teste. Aplicar transformações aprendidas só no treino. Não reportar métricas não executadas. Quando comparar modelos, mostrar F1, Recall e Precision e investigar confusões e curvas. Ajustar escolhas ao problema em vez de impor algoritmos ou hiperparâmetros fixos. API, MLflow e Docker somente quando agregarem valor e forem implementados e testados.
