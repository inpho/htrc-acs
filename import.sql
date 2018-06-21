.open alignment.sqlite

create table htrc (htrc text, record text, oclc text, lccn text);
.separator "\t"
.import /home/jaimie/alignment-htrc.tsv htrc

create table loc (lccn text, oclc text, lcco text);
.separator " "
.import /home/jaimie/alignment.tsv loc

CREATE INDEX loc_lccn ON loc (lccn);
CREATE INDEX htrc_lccn ON htrc (lccn);
CREATE INDEX loc_oclc ON loc (oclc);
CREATE INDEX htrc_oclc ON htrc (oclc);

-- process the crosswalk

CREATE TABLE crosswalk (htrc text, lcco text);
.separator " "
.import crosstab.tsv crosswalk


CREATE TABLE lcco (lcco text, title text);
.separator "\t"
.import lcco_titles.tsv lcco

SELECT crosswalk.lcco, count(*) AS htcount, lcco.title FROM crosswalk JOIN lcco ON crosswalk.lcco = lcco.lcco GROUP BY crosswalk.lcco ORDER BY htcount DESC LIMIT 10;

