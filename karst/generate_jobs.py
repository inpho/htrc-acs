from glob import glob
import subprocess

for _, inifile in zip(range(10), glob('data/*.htids.txt.ini')):
    area = inifile.replace('data/', '').replace('.htids.txt.ini', '')
    print(f'SUBMITTING {area}')
    subprocess.call(f'bash submit_qsub.sh {area}', shell=True)

