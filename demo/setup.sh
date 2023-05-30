pip install --user -r requirements.txt
python src/main.py clean -vt
python src/main.py build --vectorizers=doc2vec-200 --sias=linear --similarities=euclidean -vt