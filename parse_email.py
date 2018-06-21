from glob import glob
import re

PATTERN = '/home/jaimie/Desktop/karst/*.eml'
def get_seconds(resource):
    CPUT_PATTERN = f'resources_used.{resource}=(?P<hours>\d*):(?P<minutes>\d*):(?P<seconds>\d*)'
    
    seconds = 0
    for filename in glob(PATTERN): 
        with open(filename) as email:
            email = email.read()
            cput = re.search(CPUT_PATTERN, email)
            if cput:
                seconds += int(cput.group('seconds'))
                seconds += 60 * int(cput.group('minutes'))
                seconds += 60 * 60 * int(cput.group('hours'))

    return seconds

print(get_seconds('cput')/ 60 / 60, "hours of cpu time")
print(get_seconds('cput')/ 60 / 60 / 24, "days of cpu time")
print(get_seconds('walltime')/ 60 / 60, "hours of wall time")
print(get_seconds('walltime')/ 60 / 60 / 24, "days of wall time")

print(get_seconds('cput') / 211967, "cpu-seconds/model")
print(get_seconds('walltime') / 211967, "wall-seconds/model")

