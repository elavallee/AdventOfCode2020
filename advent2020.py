from itertools import combinations
from collections import defaultdict
from functools import lru_cache
import re
import pdb
import cProfile

def oneA(inp, r=2):
    for num in combinations(inp, r):
        if sum(num) == 2020: return prod(num)

def prod(nums):
    "Return the product of all the numbers in nums."
    result = 1
    for num in nums:
        result *= num
    return result

inp = [int(x) for x in """1721
979
366
299
675
1456""".splitlines()]

print(oneA(inp))

print(oneA(inp, 3))

inp = [int(x) for x in open('inp1.txt').readlines()]

print(oneA(inp))
print(oneA(inp, 3))

def isValid(minLet, maxLet, let, pssword) -> bool:
    cnt = pssword.count(let)
    return cnt >= minLet and cnt <= maxLet

def twoA(inps, isValid=isValid):
    minLets, maxLets, lets, psswords = parseInps(inps)
    cnt = 0
    for ix in range(len(psswords)):
        if isValid(minLets[ix], maxLets[ix], lets[ix], psswords[ix]): cnt += 1
    return cnt

def parseInps(inps):
    minLets, maxLets, lets, psswords = [], [], [], []
    for inp in inps:
        digits = re.findall(r'\d+', inp)
        minLets.append(int(digits[0]))
        maxLets.append(int(digits[1]))
        mtch = re.search(r'[a-z]:', inp)
        lets.append(mtch.group(0)[0])
        mtch = re.search(r':\s+([a-z]+)', inp)
        psswords.append(mtch.group(1))
    return minLets, maxLets, lets, psswords

def isValid2a(pos1, pos2, let, pssword) -> bool:
    return ((pssword[pos1-1] == let and pssword[pos2-1] != let) or
            (pssword[pos1-1] != let and pssword[pos2-1] == let))

inps = """1-3 a: abcde
1-3 b: cdefg
2-9 c: ccccccccc""".splitlines()

print(twoA(inps))
print(twoA(inps, isValid2a))

inps = open('inp2.txt').readlines()

print(twoA(inps))
print(twoA(inps, isValid2a))

def threeA(inps, m=3, n=1):
    accum, trees = m, 0
    for ix in range(n, len(inps), n):
        inp = inps[ix]
        if inp[accum % len(inp)] == '#':
            trees += 1
        accum += m
    return trees

inps = """..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#""".splitlines()

print(threeA(inps))

print(threeA(inps, 1, 1)*threeA(inps, 3, 1)*threeA(inps, 5, 1)*threeA(inps, 7, 1)*threeA(inps, 1, 2))
        
inps = [x.strip() for x in open('inp3.txt').readlines()]

print(threeA(inps))

print(threeA(inps, 1, 1)*threeA(inps, 3, 1)*threeA(inps, 5, 1)*threeA(inps, 7, 1)*threeA(inps, 1, 2))

def fourA(inps):
    valid = 0
    for inp in inps:
        pssport = getPassport(inp)
        valid += 1 if isValidPassport(pssport) else 0
    return valid

def readBatchFile(txt) -> list:
    inps, tmp = [], ''
    for line in txt.splitlines():
        if line.strip() == '':
            inps.append(tmp.strip())
            tmp = ''
        else:
            tmp += ' ' + line.strip()
    inps.append(tmp.strip())
    return inps

def getPassport(inp) -> dict:
    "Return a dict of {field: value, ...} for a passport."
    txt = inp.replace(' ', ', ')
    txt = re.sub(r'((?:\w|#)+)', r"'\1'", txt) # convert everything to str's
    return eval('{' + txt + '}')

def isValidPassport(passport) -> bool:
    "Part A."
    fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
    passportFeilds = set(passport.keys())
    return fields - passportFeilds == set()

def isValidPassport(p) -> bool:
    "Part B."
    fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
    ecls   = {'amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'}
    passportFeilds = set(p.keys())
    if fields - passportFeilds != set(): return False
    hgt = p['hgt']
    cnd = (150 <= int(hgt[:-2]) <= 193 if hgt.endswith('cm') else
           59  <= int(hgt[:-2]) <= 76  if hgt.endswith('in') else
           False)
    return (1920 <= int(p['byr']) <= 2002           and
            2010 <= int(p['iyr']) <= 2020           and
            2020 <= int(p['eyr']) <= 2030           and
            cnd                                     and
            re.fullmatch(r'#[0-9a-f]{6}', p['hcl']) and
            p['ecl'] in ecls                        and
            re.fullmatch(r'\d{9}', p['pid']))

txt = """ecl:gry pid:860033327 eyr:2020 hcl:#fffffd
byr:1937 iyr:2017 cid:147 hgt:183cm

iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884
hcl:#cfa07d byr:1929

hcl:#ae17e1 iyr:2013
eyr:2024
ecl:brn pid:760753108 byr:1931
hgt:179cm

hcl:#cfa07d eyr:2025 pid:166559648
iyr:2011 ecl:brn hgt:59in"""

print(readBatchFile(txt))
print(getPassport('ecl:gry pid:860033327 eyr:2020 hcl:#fffffd'))
print(fourA(readBatchFile(txt)))
print(fourA(readBatchFile(open('inp4.txt').read())))

def fiveA(inps):
    IDs = [readBoard(inp)[2] for inp in inps]
    return max(IDs)

def readBoard(inp):
    rows, cols = list(range(128)), list(range(8))
    for char in inp[:7]:
        half = len(rows) // 2
        rows = rows[:half] if char == 'F' else rows[half:]
    for char in inp[7:]:
        half = len(cols) // 2
        cols = cols[:half] if char == 'L' else cols[half:]
    return rows[0], cols[0], rows[0] * 8 + cols[0]

def fiveB(inps):
    IDs = set([readBoard(inp)[2] for inp in inps])
    missings = getMissing(IDs)
    for missing in missings:
        if missing - 1 in IDs and missing + 1 in IDs:
            return missing

def getMissing(IDs):
    allIDs = set(range(127*8 + 7))
    return allIDs - IDs

print(readBoard('FBFBBFFRLR'))
print(readBoard('BFFFBBFRRR'))
print(readBoard('FFFBBBFRRR'))
print(readBoard('BBFFBBFRLL'))

inps = [inp.strip() for inp in open('inp5.txt').readlines()]

print(fiveA(inps))

print(fiveB(inps))

def sixA(inps):
    cnts = [len(set(''.join(inp.split()))) for inp in inps]
    return sum(cnts)

def sixB(inps):
    cnts = [len(intersect([set(x) for x in inp.split()])) for inp in inps]
    return sum(cnts)

def intersect(items) -> set:
    return items[0].intersection(*items[1:])

inps = readBatchFile("""abc

a
b
c

ab
ac

a 
a
a
a

b""")

print(sixA(inps))
print(sixB(inps))

inps = readBatchFile(open('inp6.txt').read())

print(sixA(inps))
print(sixB(inps))

def readBagLine(line):
    halves = line[:-1].split(' bags contain ') # split and also remove the `.`
    container  = halves[0]
    tmpcontainees = halves[1].split(', ')
    containees = []
    for containee in tmpcontainees:
        containees.append((int(containee[0]) if containee[0].isdigit() else 0,
                           containee[2:-5] if containee.endswith('s') else
                           containee[2:-4]))
    return container, containees

def sevenA(inps) -> int:
    bags = defaultdict(set)
    for inp in inps:
        container, containees = readBagLine(inp)
        containeeBags = [x[1] for x in containees]
        for bag in containeeBags:
            if bag != ' other':
                bags[bag].add(container)
    return len(getUniqueBags(bags))

def getUniqueBags(bags, bag='shiny gold', uniqueBags=None) -> set:
    if uniqueBags is None: uniqueBags = set()
    if bag in bags:
        for b in bags[bag]:
            uniqueBags.add(b)
            uniqueBags = getUniqueBags(bags, b, uniqueBags)
        return uniqueBags
    else:
        return uniqueBags

def sevenB(inps) -> int:
    bags = defaultdict(list)
    for inp in inps:
        container, containees = readBagLine(inp)
        bags[container].append(containees)
    return getBagCnt(bags)

def getBagCnt(bags, bag='shiny gold', mult=1) -> int:
    cnt = 0
    if bag in bags:
        for l in bags[bag]:
             for c, b in l:
                 cnt += c * mult
                 cnt += getBagCnt(bags, b, c * mult)
        return cnt
    else:
        return cnt

print(readBagLine('light red bags contain 1 bright white bag, 2 muted yellow bags.'))

inps = """light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags.""".splitlines()

print(sevenA(inps))

inps = [x.strip() for x in open('inp7.txt').readlines()]

print(sevenA(inps))

inps = """shiny gold bags contain 2 dark red bags.
dark red bags contain 2 dark orange bags.
dark orange bags contain 2 dark yellow bags.
dark yellow bags contain 2 dark green bags.
dark green bags contain 2 dark blue bags.
dark blue bags contain 2 dark violet bags.
dark violet bags contain no other bags.""".splitlines()

print(sevenB(inps))

inps = [x.strip() for x in open('inp7.txt').readlines()]

print(sevenB(inps))

accumulator = 0
linenos = set()
terminated = False

def eightA(program, lineno=0) -> int:
    global accumulator, terminated
    cmd, cnt = parseProgLine(program[lineno])
    if lineno == len(program) - 1:
        terminated = True
        if cmd == 'acc': accumulator += cnt
        return accumulator
    if lineno in linenos: return accumulator
    else:
        linenos.add(lineno)
        if cmd == 'nop': return eightA(program, lineno + 1)
        if cmd == 'acc':
            accumulator += cnt
            return eightA(program, lineno + 1)
        if cmd == 'jmp': return eightA(program, lineno + cnt)

def parseProgLine(line):
    return line[:3], int(line[4:])

def eightB(program) -> int:
    nops = [ix for ix in range(len(program)) if program[ix][:3] == 'nop']
    jmps = [ix for ix in range(len(program)) if program[ix][:3] == 'jmp']
    cnt = eightBhelper(program, nops, 'nop', 'jmp')
    if terminated: return cnt
    cnt = eightBhelper(program, jmps, 'jmp', 'nop')
    return cnt

def eightBhelper(program, ixs, repl1, repl2):
    global accumulator, terminated, linenos
    for ix in ixs:
        accumulator, linenos, terminated = 0, set(), False
        altprogram = list(program)
        altprogram[ix] = altprogram[ix].replace(repl1, repl2)
        cnt = eightA(altprogram)
        if terminated: return cnt
    return None

program = """nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6""".splitlines()

print(eightA(program))
print(eightB(program))

program = open('inp8.txt').read().splitlines()

accumulator = 0
linenos = set()
terminated = False

print(eightA(program))
print(eightB(program))

def nineA(inps, num=25):
    inps = [int(inp) for inp in inps]
    for ix, inp in enumerate(inps[num:]):
        inpIx = ix + num
        nums = {sum(set(x)) for x in combinations(inps[inpIx - num:inpIx], 2)}
        if inp not in nums: return inp

def nineB(inps, num=675280050):
    inps = [int(inp) for inp in inps]
    for start in range(len(inps)):
        for last in range(start+2, len(inps)):
            sub = inps[start:last]
            tot = sum(sub)
            if tot == num: return min(sub) + max(sub)
            elif tot > num: break

inps = """35
20
15
25
47
40
62
55
65
95
102
117
150
182
127
219
299
277
309
576""".splitlines()

print(nineA(inps, 5))
print(nineB(inps, 127))

inps = open('inp9.txt').read().splitlines()

print(nineA(inps))
print(nineB(inps))

def tenA(inps) -> int:
    inps  = sorted([0] + [int(x) for x in inps])
    diffs = diff(inps)
    return diffs.count(1) * (diffs.count(3) + 1)

def tenB(inps):
    inps  = sorted([0] + [int(x) for x in inps])
    diffs = diff(inps)
    diffs = ''.join([str(x) for x in diffs])
    ones = re.findall(r'1+', diffs)
    cnts = [a(len(x))+1 for x in ones]
    return prod([cnt for cnt in cnts if cnt != 0])

@lru_cache()
def a(n) -> int:
    if n == 2: return 1
    elif n == 1: return 0
    elif n == 0: return 0
    else: return a(n-1) + a(n-2) + a(n-3) + 2

# Brute force method below
"""def tenB(inps):
    inps  = sorted([0] + [int(x) for x in inps])
    diffs = diff(inps)
    ixs   = [ix for ix in range(len(diffs)) if diffs[ix] == 1]
    cnt = 1
    for n in range(1, len(ixs) + 1):
        for tup in combinations(ixs, n):
            possible = list(inps)
            for x in sorted(tup, reverse=True):
                del possible[x + 1]
            if inps[-1] in possible and all([1 <= d <= 3 for d in diff(possible)]):
                cnt += 1
    return cnt"""
            
def diff(inps) -> list:
    return [inps[ix] - inps[ix - 1] for ix in range(1, len(inps))]

inps = """16
10
15
5
1
11
7
19
6
12
4""".splitlines()

print(tenA(inps))
print(tenB(inps))

inps = """28
33
18
42
31
14
46
20
48
47
24
23
49
45
19
38
39
11
1
32
25
35
8
17
7
9
4
2
34
10
3""".splitlines()

print(tenA(inps))
print(tenB(inps))

inps = open('inp10.txt').read().splitlines()

print(tenA(inps))
print(tenB(inps))

def elevenA(inps):
    pass
