import argparse
import numpy as np
import random
import readline
import sys

"""
    Master the fundamentals of mathematics by practicing the 
    multiplication & division tables, percentages, and fractions
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


def printBar(l=50):
    print('*'*l)
    

def printError(msg):
    print(f'{red}{msg}{rst}')


def getValues(problemSet, maxValue=15):
    a, b = [], []
    if problemSet == 'multiplication' or problemSet == 'division':
        a = [v for v in range(3, maxValue+1, 1) if v not in [10]]
        b = [v for v in range(3, 17, 1)]
    elif problemSet == 'percentage':
        a = [v for v in range(100, 400, 50)]
        b = [v / 100 for v in range(10, 100, 10)]
    # print(f'a: {a}\nb: {b}')
    return a, len(str(a[-1])), b, len(str(b[-1]))


def multiplication(hard=False):
    print('Multiplication Tables:')
    val, lenV, num, lenN = getValues(problemSet='multiplication', maxValue=N)
    for _ in range(len(val)-1):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {pink}{v}{rst}')

        # Test
        if hard:
            random.shuffle(num)
        for n in num:
            value = v * n
            spaceN = " " * (lenN - len(str(n)))
            while True:
                x = input(f'  {spaceV}{v} x {spaceN}{n} = ')
                if x:
                    if float(x) == value:
                        break
                    else:
                        deleteLine()
                        printError(f'  {spaceV}{v} x {spaceN}{n} = {x}')
                else:
                    deleteLine()
    printBar()


def division(hard=False):
    print('Division Tables:')
    div, lenD, num, _ = getValues(problemSet='division', maxValue=N)
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
                        printError(f'  {spaceN}{n} / {d}{spaceD} = {x}')
                else:
                    deleteLine()
    printBar()


def percentage(hard=False):
    print('Percentage:')
    val, lenV, per, lenP = getValues(problemSet='percentage', maxValue=N)
    for _ in range(len(val)-1):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {pink}{v}{rst}')

        # Determine values
        percents = [p for p in per if (v * p) % 1 == 0]
        # print(f'Per: {pink}{percents}{rst}')
        if hard:
            random.shuffle(percents)

        # Test
        for p in percents:
            pInt = int(p*100)
            value = v * p
            spaceP = " "
            while True:
                x = input(f'  {pInt}% of {v} = ')
                if x:
                    if float(x) == round(value, 2):
                        break
                    else:
                        deleteLine()
                        printError(f'  {pInt}% of {v} = {value}')
                else:
                    deleteLine()


def question():
    tables = {'1': 'multiply', '2': 'divide', '3': 'percentage'}
    print('Select Exercise:')
    for k, v in tables.items():
        print(f'{k}: {v}')
    x = input('Enter value: ')
    if x in tables.keys() or x in tables.values():
        printBar()
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
        elif exercise == 'percentage':
            percentage(randomize)
