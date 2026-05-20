import argparse
import ast
import json
import pickle


def parse_items(value):
    """Return a clean list[str] from a list, tuple, JSON/list string, or comma string."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text[0] in "[{(":
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = [text]
        else:
            value = [part.strip() for part in text.split(",")]

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    cleaned = []
    seen = set()
    for item in value:
        item_str = str(item).strip()
        if item_str and item_str not in seen:
            cleaned.append(item_str)
            seen.add(item_str)
        if len(cleaned) == 10:
            break

    return cleaned


def parse_customer_id(customer_id):
    text = str(customer_id).strip()
    if not text:
        raise ValueError("empty customer_id")
    return int(text)


def fix_pkl_format(input_path, output_path):
    with open(input_path, "rb") as f:
        raw_data = pickle.load(f)

    if isinstance(raw_data, str):
        raw_data = ast.literal_eval(raw_data)

    if not isinstance(raw_data, dict):
        raise TypeError(f"Expected dict, got {type(raw_data).__name__}")

    fixed = {}
    skipped = 0

    for customer_id, items in raw_data.items():
        try:
            fixed[parse_customer_id(customer_id)] = parse_items(items)
        except (TypeError, ValueError):
            skipped += 1

    with open(output_path, "wb") as f:
        pickle.dump(fixed, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {output_path}")
    print(f"Customers: {len(fixed)}")
    print(f"Skipped invalid customers: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result.pkl")
    parser.add_argument("--output", default="result_fixed.pkl")
    args = parser.parse_args()

    fix_pkl_format(args.input, args.output)


if __name__ == "__main__":
    main()
