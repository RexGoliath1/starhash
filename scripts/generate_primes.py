from sympy import nextprime
import numpy as np

count = 40
base = np.array([2**n for n in range(0, count)])

primes = []
for num in base:
    primes.append(nextprime(num))

# These prime numbers should be stored in config.yaml in order to pick the catalog size 
# Hash distribution is most effective when the modulo is prime
print(primes)
