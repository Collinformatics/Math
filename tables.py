import argparse
import numpy as np
import random
import readline
import sys

"""
    Master fundamentals the of mathematics by practicing the 
    multiplication & division tables
"""

parser = argparse.ArgumentParser(description='Mathematics Tables')
parser.add_argument('-r', '--randomize', action='store_true',
                       help='Increase difficulty by randomizing the order of values')
parser.add_argument('-m', '--maxvalue', type=int, default=15,
                    help='Maximal value for the table')
args = parser.parse_args()
randomize = args.randomize
N = args.maxvalue
if N <= 3:
    print(f'Max Value (-m / --maxvalue) cannot be ≤ 3\n'
          f'Resetting to 4')
    N = 4

# Colors
red = '\033[31m'
pink = '\033[35m'
rst = '\033[0m'


def deleteLine(nLines=1):
    print('\033[F\033[2K' * nLines, end='') # Move cursor up & delete


def pBar(l=50):
    print('*'*l)


def getValues(maxValue=15):
    a = [v for v in range(3, maxValue+1, 1)]
    b = [v for v in range(3, 17, 1)]

    return a, len(str(a[-1])), b, len(str(b[-1]))


def multiplication(hard=False):
    print('Multiplication Tables:')
    val, lenV, col, lenC = getValues(maxValue=N)
    for _ in range(len(val)-1):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {pink}{v}{rst}')

        # Test
        if hard:
            random.shuffle(col)
        for c in col:
            value = v * c
            spaceC = " " * (lenC - len(str(c)))
            while True:
                x = input(f'  {spaceV}{v} x {spaceC}{c} = ')
                if x:
                    if float(x) == value:
                        break
                    else:
                        deleteLine()
                        print(f'  {red}{spaceV}{v} x {spaceC}{c} = {x}{rst}')
                else:
                    deleteLine()
    pBar()


def division(hard=False):
    print('Division Tables:')
    div, lenD, num, _ = getValues(maxValue=N)
    for _ in range(len(div)-1):
        # Select value
        i = random.randint(1, len(div)-1)
        d = div[i]
        div = np.delete(div, i)
        spaceD = " " * (lenD - len(str(d)))
        print(f'\nValue: {pink}{d}{rst}')

        # Determine values
        if hard:
            random.shuffle(num)
        values = [d * n for n in num]
        lenN = len(str(values[-1]))

        # Test
        for n in values:
            value = n / d
            if value == 1.0 or int(value) != value:
                # print(f'* {d}, {n}, {value}')
                continue
            spaceN = " " * (lenN - len(str(n)))
            while True:
                x = input(f'  {spaceN}{n} / {d}{spaceD} = ')
                if x:
                    if float(x) == round(value, 2):
                        break
                    else:
                        deleteLine()
                        print(f'  {red}{spaceN}{n} / {d}{spaceD} = {x}{rst}')
                else:
                    deleteLine()
    pBar()


def question():
    tables = {'1': 'multiply', '2': 'divide'}
    print('Select Exercise:')
    for k, v in tables.items():
        print(f'{k}: {v}')
    x = input('Enter value: ')
    if x in tables.keys() or x in tables.values():
        pBar()
        return tables[x]
    else:
        deleteLine(4)
        return question()


if __name__ == '__main__':
    for _ in range(100):
        # exercise = 'multiply'
        exercise = question()
        if exercise == 'multiply':
            multiplication(randomize)
        elif exercise == 'divide':
            division(randomize)
