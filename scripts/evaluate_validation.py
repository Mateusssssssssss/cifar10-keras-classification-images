from cifar10_classifier.evaluate import evaluate_validation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    print(evaluate_validation(args.config))
