#!/bin/bash
# Software dependencies: wget (https://www.gnu.org/software/wget).

# Create directories
mkdir "../data/external/"
mkdir "../data/input/"

# Download files
wget -O "../data/external/PLOS_2015-2017_idArt-DOI-PY-Journal-Title-LargerDiscipline-Discipline-Specialty.csv"  "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/3CS5ES/NTNXEX&format=original"
wget -O "../data/input/altmetric_counts.csv"  "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/3CS5ES/EX5YRG&format=original"
wget -O "../data/input/graph_api_counts.csv" "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/3CS5ES/DNBO4X&format=original"
wget -O "../data/input/plos_one_articles.csv"  "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/3CS5ES/Q6QML0&format=original"
wget -O "../data/input/query_details.csv"  "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/3CS5ES/FTC8BT&format=original"
