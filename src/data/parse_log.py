import re

# 
def parse_log():
    pass

# This takes all the lines from the log of a battle between "|showteam|"(inclusive) and "|turn|1" (exclusive) and returns it
# 
def extractStartingInformation(log:str) -> str:
    return log[log.find('|showteam|'):log.find('|turn|1')]

def getLeadPokemon(log) -> list:
    return [
        findSubstring(r"(p1a:[^|]*)", log),
        findSubstring(r"(p1b:[^|]*)", log),
        findSubstring(r"(p2a:[^|]*)", log),
        findSubstring(r"(p2b[^|]*)", log),
        ]

def findSubstring(regex, log) -> str: 
    search = re.search(regex, log)   
    result = search.group(1) if search else ""
    return result

with open("data\\raw\\samplelog.txt", 'r') as file:
    for line in file:
        shortLog = extractStartingInformation(line)
        print(shortLog+'\n\n')

        leadPokemon = getLeadPokemon(shortLog)
        print(f'\n\n {leadPokemon}')

