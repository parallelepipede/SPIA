from typing import *
import random


def list_str(L: List) -> str:
    return "[" + ", ".join(str(elt) for elt in L) + "]"


def list_rotate(L: List, n: int) -> List:
    assert n in range(len(L))
    return L[n:] + L[:n]


def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    if len(a) != len(b):
        return False
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched


def maybe() -> bool:
    return random.choice([True, False])


# class A:
#     def __init__(self):
#         self.a = 0
#         self.b = 8
#
#     def itself(self) -> __main__.A:
#         print(type(self))
#         return self
