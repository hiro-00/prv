import sys
import re

class CharacterDiff():
    def gen(self, s1, s2):
        return [len(s1) / len(s2)]


class WordDiff():
    def gen(self, s1, s2):
        return [len(s1.split()) / len(s2.split())]

class DigitDiff():
    def gen(self, s1, s2):
        return [len(re.findall(r"\d", s1)) / len(re.findall(r"\d", s2))]

def main(*argv):
    pass


if __name__ == "__main__":
    main(sys.argv[1])