import yaml

def main():
    with open("/Users/alexwang/BoundBench/bound_bench/scripts/simply.yaml", "r") as f:
        params = yaml.safe_load(f)

    print("Training model with the following parameters:")
    print(params)
    # Add training logic here

if __name__ == "__main__":
    main()
