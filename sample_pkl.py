import argparse
import pickle
from itertools import islice
from pprint import pprint


def show_samples(path, n):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    print(f"File: {path}")
    print(f"Type: {type(data).__name__}")
    print(f"Total customers: {len(data)}")
    print(f"Samples: {n}")
    print()

    samples = dict(islice(data.items(), n))
    pprint(samples, sort_dicts=False)
    print()

    for customer_id, items in samples.items():
        print(
            f"customer_id={customer_id!r} ({type(customer_id).__name__}), "
            f"items_type={type(items).__name__}, "
            f"items_count={len(items) if hasattr(items, '__len__') else 'unknown'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./result_v11.pkl")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    show_samples(args.path, args.n)


if __name__ == "__main__":
    main()
