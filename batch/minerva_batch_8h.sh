#!/bin/bash
source ~/anaconda3/bin/activate minerenv #correct
cd /home/jarvis/codes/Minerva/ #correct

mv ./batch/reports/reports.log ./batch/reports/reports.log.old

cp ./database/Economics.db ./database/Economics.db.backup
python ./batch/economics_db.py

python ./batch/sentiment.py
python ./batch/kr_ecos.py
python ./batch/kr_marks.py
python ./batch/jp_ecos.py
python ./batch/in_ecos.py
python ./batch/global_.py
python ./batch/cn_ecos.py
python ./batch/de_ecos.py
python ./batch/us_ecos.py
python ./batch/us_marks.py
python ./batch/global_derivatives.py

python ./batch/email_.py

conda deactivate
