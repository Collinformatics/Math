import argparse
import numpy as np
import random
import readline
import sys

"""
    Master the mathematics fundamentals by practicing the multiplication & division tables
"""

parser = argparse.ArgumentParser(description='Mathematics Tables')
parser.add_argument('-d', '--difficult', action='store_true',
                       help='Increase difficulty by skipping the 5s and 10s')
parser.add_argument('-m', '--maxvalue', type=int, default=15,
                    help='Maximal value for the table')
args = parser.parse_args()
hardMode = args.difficult
N = args.maxvalue

# Colors
red = '\033[31m'
pink = '\033[35m'
rst = '\033[0m'


def deleteLine(nLines=1):
    print('\033[F\033[2K' * nLines, end='') # Move cursor up & delete


def getValues(valueSet, maxValue=15, hard=False):
    a = [v for v in range(3, maxValue+1, 1)]
    b = [v for v in range(3, 17, 1)]
    if valueSet == 'multiply':
        if hard:
            # Remove 5 & 10
            a = a[(a != 5) & (a != 10)]
            b = b[(b != 5) & (b != 10)]

    return a, len(str(a[-1])), b, len(str(b[-1]))


def multiplication():
    print('Multiplication Tables:')
    val, lenV, col, lenC = getValues(valueSet='multiply', maxValue=N, hard=hardMode)
    for _ in range(len(val)-1):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {pink}{v}{rst}')

        # Test
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
    print('\nDone\n')


def division():
    print('Division Tables:')
    div, lenD, num, _ = getValues(valueSet='divide', maxValue=N, hard=hardMode)
    for _ in range(len(div)-1):
        # Select value
        i = random.randint(1, len(div)-1)
        d = div[i]
        div = np.delete(div, i)
        spaceD = " " * (lenD - len(str(d)))
        random.shuffle(num)
        factors = [d * n for n in num]
        lenN = len(str(factors[-1]))
        # if not any(n % d == 0 and n/d != 1.0  for n in num):
        #     continue
        print(f'\nValue: {pink}{d}{rst}')

        # Test
        for n in factors:
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
    print('\nDone\n')


def question():
    tables = {'0': 'multiply', '1': 'divide'}
    print('Select exercise:\n'
          '0: Multiplication\n'
          '1: Division')
    x = input('Enter value: ')
    if x in tables.keys():
        return tables[x]
    else:
        deleteLine()(4)
        return question()


if __name__ == '__main__':
    for _ in range(100):
        # exercise = 'multiply'
        exercise = question()
        if exercise == 'multiply':
            multiplication()
        elif exercise == 'divide':
            division()
