import argparse
import numpy as np
import random
import readline
import sys

"""
    Master the fundamentals of mathematics by practicing the 
    multiplication & division tables, percentages, and fractions
"""

parser = argparse.ArgumentParser(description='Mathematics Drills')
parser.add_argument('-r', '--randomize', action='store_true',
                       help='Increased difficulty via order randomization')
parser.add_argument('-m', '--maxvalue', type=int, default=15,
                    help='Max value for the operation tables')
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


def printBar(msg='', l=50):
    print('*'*l)
    if msg:
        print(msg)
    

def printError(msg):
    print(f'{red}{msg}{rst}')


def getValues(problemSet, maxValue=15):
    a, b = [], []
    if problemSet == 'multiplication' or problemSet == 'division':
        a = [v for v in range(3, maxValue+1, 1) if v not in [10]]
        b = [v for v in range(3, 17, 1)]
    elif problemSet == 'double digets':
        v1 = random.randint(1, 9) * 10
        v2 = random.randint(1, 9)
        a.append(v1 + v2)
        b.append(v1 + v2)
        for _ in range (2):
            v2 = random.randint(1, 9)
            b.append(v1 + v2)
        for _ in range (2):
            v1 = random.randint(1, 9) * 10
            v2 = random.randint(1, 9)
            a.append(v1 + v2)
            b.append(v1 + v2)
        random.shuffle(b)
    elif problemSet == 'percentages':
        a = [v for v in range(100, maxValue, 50)]
        b = [v / 100 for v in range(10, 100, 10)]
    elif problemSet == 'fractions':
        a = [v for v in range(1, maxValue, 1)]
        b = [v / 100 for v in range(1, maxValue, 1)]
    # print(f'a: {a}\nb: {b}')
    return a, len(str(a[-1])), b, len(str(b[-1]))


def multiplication(shuffle=False):
    print('Multiplication Tables:')
    val, lenV, num, lenN = getValues(problemSet='multiplication', maxValue=N)
    nRounds = len(val) - 1
    for r in range(1, nRounds+1):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        spaceV = " " * (lenV - len(str(v)))
        print(f'\nValue: {pink}{v}{rst} ({r}/{nRounds})')

        # Test
        if shuffle:
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


def multiplicationDoubleDiget(shuffle=False):
    printBar(msg='Double Diget Multiplication:')
    valA, lenA, valB, lenB = getValues(problemSet='double digets', maxValue=N)
    nRounds = len(valA) - 1
    for r in range(1, nRounds+1):
        # Select value
        i = random.randint(1, len(valA)-1)
        vA = valA[i]
        valA = np.delete(valA, i)
        spaceA = " " * (lenA - len(str(vA)))
        print(f'\nValue: {pink}{vA}{rst} ({r}/{nRounds})')

        # Test
        for vB in valB:
            value = vA * vB
            if value == 1.0 or int(value) != value:
                # print(f'* {d}, {n}, {value}')
                continue
            spaceB = " " * (lenA - len(str(vB)))
            while True:
                x = input(f'  {spaceA}{vA} * {vB}{spaceB} = ')
                if x:
                    if float(x) == round(value, 2):
                        break
                    else:
                        deleteLine()
                        printError(f'  {spaceA}{vA} * {vB}{spaceB} = {x}')
                else:
                    deleteLine()
    printBar()


def division(shuffle=False):
    printBar(msg='Division Tables:')
    div, lenD, num, _ = getValues(problemSet='division', maxValue=N)
    nRounds = len(div) - 1
    for r in range(1, nRounds+1):
        # Select value
        i = random.randint(1, len(div)-1)
        d = div[i]
        div = np.delete(div, i)
        spaceD = " " * (lenD - len(str(d)))
        print(f'\nValue: {pink}{d}{rst} ({r}/{nRounds})')

        # Determine values
        if shuffle:
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


def percentages(shuffle=False):
    printBar(msg='Percentages:')
    val, _, per, _ = getValues(problemSet='percentages', maxValue=1000)
    nRounds = len(val) - 1
    for r in range(nRounds):
        # Select value
        i = random.randint(1, len(val)-1)
        v = val[i]
        val = np.delete(val, i)
        print(f'\nValue: {pink}{v}{rst} ({r}/{nRounds})')

        # Determine values
        percents = [p for p in per if (v * p) % 1 == 0]
        # print(f'Per: {pink}{percents}{rst}')
        if shuffle:
            random.shuffle(percents)

        # Test
        for p in percents:
            pInt = int(p*100)
            value = v * p
            while True:
                x = input(f'  {pInt}% of {v} = ')
                if x:
                    if float(x) == round(value, 2):
                        break
                    else:
                        deleteLine()
                        printError(f'  {pInt}% of {v} = {x}')
                else:
                    deleteLine()


def fractions(shuffle=False):
    printBar(msg='Fractions:')
    val, lenV, div, lenD = getValues(problemSet='fractions', maxValue=10)
    nRounds = len(val) - 1


def question():
    tables = {
        '1': 'Multiplication Tables',
        '2': 'Double Diget Multiplication',
        '3': 'Division Tables',
        '4': 'Percentages',
        '5': 'Fractions'
    }
    print('Select Exercise:')
    for k, v in tables.items():
        print(f'  {k}: {v}')
    return tables[input('Enter value: ')]


if __name__ == '__main__':
    while True:
        # exercise = 'multiply'
        exercise = question()
        if exercise == 'Multiplication Tables':
            multiplication(randomize)
        elif exercise == 'Double Diget Multiplication':
            multiplicationDoubleDiget()
        elif exercise == 'Division Tables':
            division(randomize)
        elif exercise == 'Percentages':
            percentages(randomize)
        elif exercise == 'Fractions':
            fractions(randomize)
        else:
            break
