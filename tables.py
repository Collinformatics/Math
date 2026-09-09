import argparse
import numpy as np
import random
import readline

"""
    Master the mathematics fundamentals by practicing the multiplication & division tables
"""

parser = argparse.ArgumentParser(description='Mathematics Tables')
parser.add_argument('-d', '--difficult', action='store_true',
                       help='Increase difficulty by skipping the 5s and 10s')
args = parser.parse_args()
hardMode = args.difficult

# Colors
red = '\033[31m'
pink = '\033[35m'
rst = '\033[0m'


def delLine(num=1):
    print('\033[F\033[2K' * num, end='') # Move cursor up & delete


def getValues(N=15, set, hard=False):
    a = np.arange(3, N+1)
    if set == 'multiply':
        b = np.arange(3, 16)
        if hard:
        # Remove 5 & 10
        a = a[(a != 5) & (a != 10)]
        b = b[(b != 5) & (b != 10)]
    elif set == 'divide':
        b = [100, 50, 10]

    return a, len(str(a[-1])), b, len(str(b[-1]))


def multiplication():
    print('Multiplication Tables:')
    val, lenV, col, lenC = getValues(N=15, set='multiply', hard=hardMode)
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
                if float(x) == value:
                    break
                else:
                    delLine()
                    print(f'  {red}{spaceV}{v} x {spaceC}{c} = {x}{rst}')
    print('\nDone\n')


def division():
    print('Division Tables:')
    divisor, lenD, numerator, lenN = getValues(N=15, hard=hardMode)
    for _ in range(len(divisor)-1):
        # Select value
        i = random.randint(1, len(val)-1)
        d = val[i]
        val = np.delete(divisor, i)
        spaceD = " " * (lenD - len(str(d)))
        print(f'\nValue: {pink}{d}{rst}')

        # Test
        for n in numerator:
            value = n / d
            spaceN = " " * (lenN - len(str(n)))
            while True:
                x = input(f'   {spaceN}{n} / {d}{spaceD} = ')
                if float(x) == value:
                    break
                else:
                    delLine()
                    print(f'  {red}{spaceN}{n} / {d}{spaceD} = {x}{rst}')
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
        delLine(4)
        return question()


if __name__ == '__main__':
    for _ in range(100):
        # exercise = question()
        exercise = 'multiply'
        if exercise == 'multiply':
            multiplication()
        elif exercise == 'divide':
            division()
