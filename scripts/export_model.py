"""Serializa o checkpoint congelado para inferência."""

from cifar10_classifier.serialization import export_model


if __name__ == "__main__":
    print(export_model().resolve())
