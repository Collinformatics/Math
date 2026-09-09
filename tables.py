import argparse
import numpy as np
import random

"""
    Master the mathematics fundamentals by practicing the multiplication & division tables
"""


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--multiply', action='store_false')
args = parser.parse_args()

mul = args.multiply


def multiplication():
    print('Multiplication Tables:')
    val = np.arange(3, 15)
    col = np.arange(3, 14)
    lenV = len(str(val[-1]))
    lenC = len(str(col[-1]))
    for v in val:
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {v}')
        for c in col:
            value = v * c
            spaceC = " " * (lenC - len(str(c)))
            while True:
                x = float(input(f'  {spaceV}{v} x {spaceC}{c} = '))
                if x == value:
                    break
    print('\nDone\n')


def question():
    tables = {'0': 'mul', '1': 'div'}
    print('Select exercise:\n'
          '0: Multiplication table\n'
          '1: Division table')
    x = input('Enter value: ')
    if x in tables.keys():
        return tables[x]
    else:
        print('\033[F\033[2K' * 4, end='') # Move cursor up & delete
        return question()


if __name__ == '__main__':
    table = 'mul'
    for _ in range(10**3):
        # table = question()
        if table == 'mul':
            multiplication()
            break
