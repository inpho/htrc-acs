# for each area in:
#   SELECT lcco FROM subject_counts WHERE c > 500 and c < 5000;
#   # generate list of volumes
#   # init
#   # prep
#   # train
import os
import os.path
import sqlite3

if not os.path.exists('datasets'):
    os.makedirs('datasets')

conn = sqlite3.connect('data/alignment.sqlite')
c = conn.cursor()

query = "SELECT lcco FROM subject_counts WHERE c > 500 and c < 5000;"
areas = [row[0] for row in c.execute(query)]

for area in areas:
    with open('datasets/{}.htids.txt'.format(area), 'w') as outfile:
        q2 = "SELECT htrc FROM crosswalk WHERE lcco = ?"
        outfile.write('\n'.join(
            row[0] for row in c.execute(q2, [area])))

