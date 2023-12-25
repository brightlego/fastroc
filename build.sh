python3 -m build # build the project
python3 -m pip install "$(ls *.whl | sort -V | tail -n 1)" # install the latest build
