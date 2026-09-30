import re

# 
def parse_log():
    pass

# This takes all the lines from the log of a battle between "|showteam|"(inclusive) and "|turn|1" (exclusive) and returns it
def extractStartingInformation(log:str) -> str:
    return log[log.find('|showteam|'):log.find('|turn|1')]


