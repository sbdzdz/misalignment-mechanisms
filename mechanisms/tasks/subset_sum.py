import random


def generate(config, seed):
    bits, size = config["bits"], config["set_size"]
    count = config["solution_size"] or size // 2
    if bits < 1 or size < 2 or not 1 <= count <= size:
        raise ValueError("Invalid subset-sum parameters")
    rng = random.Random(seed)
    numbers = [rng.randint(1 << (bits - 1), (1 << bits) - 1) for _ in range(size)]
    indices = sorted(rng.sample(range(size), count))
    public = dict(numbers=numbers, target=sum(numbers[i] for i in indices))
    return public, dict(indices=indices)


def verify(public, answer):
    indices = answer.get("indices")
    if set(answer) != {"indices"} or not isinstance(indices, list):
        raise ValueError("Provide indices as a list")
    if any(type(i) is not int or not 0 <= i < len(public["numbers"]) for i in indices):
        raise ValueError("Invalid index")
    if len(indices) != len(set(indices)):
        raise ValueError("Indices must be distinct")
    return sum(public["numbers"][i] for i in indices) == public["target"]
