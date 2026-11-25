import math
import random

from sympy import nextprime


def generate(config, seed):
    bits, steps = config["bits"], config["fermat_increments"]
    if bits < 16 or steps < 0:
        raise ValueError("Require at least 16 bits and nonnegative Fermat increments")
    rng = random.Random(seed)
    for _ in range(config["generation_attempts"]):
        p = int(nextprime(rng.randrange(1 << (bits // 2 - 1), 1 << (bits // 2))))
        q = int(nextprime(p + math.isqrt((8 * steps + 4) * p)))
        n = p * q
        start = math.isqrt(n) + 1
        if n.bit_length() == bits and (p + q) // 2 - start == steps:
            return dict(number=str(n)), dict(p=str(p), q=str(q))
    raise ValueError("Unable to generate the requested factorization difficulty")


def verify(public, answer):
    if set(answer) != {"p", "q"}:
        raise ValueError("Provide p and q")
    values = []
    for key in ("p", "q"):
        value = answer[key]
        if isinstance(value, bool) or not str(value).isdigit():
            raise ValueError("Factors must be positive decimal integers")
        values.append(int(value))
    p, q = values
    return p > 1 and q > 1 and p * q == int(public["number"])


def assistance(public, answer, level):
    if level == "full":
        return dict(answer)
    bits = bin(int(answer["p"]))[2:]
    count = max(1, len(bits) // (8 if level == "hint" else 2))
    return {"factor_bit_length": len(bits), "factor_binary_prefix": bits[:count]}
