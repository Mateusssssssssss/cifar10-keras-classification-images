"""Executa a avaliação final com o manifesto congelado."""

import json

from cifar10_classifier.evaluate_test import evaluate_test


if __name__ == "__main__":
    print(json.dumps(evaluate_test(), indent=2))
