from glob import glob
import subprocess

for i, inifile in zip(range(15), glob('../../data/*.htids.txt.ini')):
    #for inifile in ['TX642-840']:
    #i = 10
    if i >= 10:
        area = inifile.replace('../../data/', '').replace('.htids.txt.ini', '')
        print(f'SUBMITTING {area}')
        for j in range(10):
            subprocess.call(f'bash submit_qsub.sh {area}', shell=True)

