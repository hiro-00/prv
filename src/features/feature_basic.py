import sys
import re
from ..dist_utils import *

class CharacterDiff():
    def gen(self, s1, s2):
        a = len(s1)
        b = len(s2)
        return [abs(a-b), default_divide(a,b) if a > b else default_divide(b,a)]


class WordDiff():
    def gen(self, s1, s2):
        a = len(s1.split())
        b = len(s2.split())
        return [abs(a-b), default_divide(a,b) if a > b else default_divide(b,a)]


class DigitDiff():
    def gen(self, s1, s2):
        a = len(re.findall(r"\d", s1))
        b = len(re.findall(r"\d", s2))
        return [abs(a-b), default_divide(a,b) if a > b else default_divide(b,a)]



def main(*argv):
    pass


if __name__ == "__main__":
    main(sys.argv[1])