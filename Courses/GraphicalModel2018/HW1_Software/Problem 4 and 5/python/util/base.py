from functools import reduce

def factor_product(*args):

    return reduce(lambda phi1, phi2: phi1 * phi2, args)