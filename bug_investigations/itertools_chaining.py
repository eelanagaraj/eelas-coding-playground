#!/usr/bin/env python3

import itertools


def problem_v4(n=10):
    """ Still fails """
    d_lst = list()
    for _ in range(n):
        _temp = [0]
        d_lst = itertools.chain(d_lst, _temp)

    d_lst = list(d_lst)
    return d_lst


if __name__ == '__main__':
    results = list()
    # 45000 instantly fails
    for n in [45000, 50000]:#, 40000, 45000, 50000]:
        print(f"Beginning problem {n}")
        # res = problem(n)
        # res = problem_v1(n)
        # res = problem_v2(n)
        # res = problem_v3(n)
        res = problem_v4(n)
        results.append(res)
