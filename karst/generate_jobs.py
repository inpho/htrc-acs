from glob import glob
import subprocess

for i, inifile in zip(range(10), glob('../../data/*.htids.txt.ini')):
    if i > 0:
        area = inifile.replace('../../data/', '').replace('.htids.txt.ini', '')
        print(f'SUBMITTING {area}')
        for j in range(10):
            subprocess.call(f'bash submit_qsub.sh {area}', shell=True)

