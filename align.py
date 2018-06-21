import sqlite3
import logging

conn = sqlite3.connect('data/alignment.sqlite')
c = conn.cursor()
logging.debug("searching by oclc")
for row in c.execute('''
SELECT htrc.htrc, htrc.record, loc.lcco 
FROM loc JOIN htrc ON htrc.oclc == loc.oclc
WHERE loc.oclc != '' AND loc.lcco != ''
'''):
    print(row[0], row[1], row[2], sep='\t')

logging.debug("searching by lccn")
for row in c.execute('''
SELECT htrc.htrc, htrc.record, loc.lcco 
FROM htrc JOIN loc ON htrc.lccn == loc.lccn
WHERE htrc.lccn != '' AND loc.lcco != ''
'''):
    print(row[0], row[1], row[2], sep='\t')
