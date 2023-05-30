read -p "Are you sure you want to clean the directory? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf vectorizers models data/sentiment data/vectorized
fi