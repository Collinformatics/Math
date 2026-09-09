import numpy as np
import random
import readline

"""
    Master the mathematics fundamentals by practicing the multiplication & division tables
"""

# Colors
red = '\033[31m'
rst = '\033[0m'


def delLine(num=1):
    print('\033[F\033[2K' * num, end='')


def multiplication():
    print('Multiplication Tables:')
    N = 15
    val = np.arange(3, N+1)
    col = np.arange(3, 14)
    lenV = len(str(val[-1]))
    lenC = len(str(col[-1]))
    for _ in range(N):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {v}')

        # Test
        for c in col:
            value = v * c
            spaceC = " " * (lenC - len(str(c)))
            while True:
                x = input(f'  {spaceV}{v} x {spaceC}{c} = ')
                if float(x) == value:
                    break
                else:
                    delLine()
                    print(f'  {red}{spaceV}{v} x {spaceC}{c} = {x}{rst}')
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
        delLine(4) # Move cursor up & delete
        return question()


if __name__ == '__main__':
    table = 'mul'
    for _ in range(10**3):
        # table = question()
        if table == 'mul':
            multiplication()
            break
