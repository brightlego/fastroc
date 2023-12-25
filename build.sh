python3 -m build # build the project
python3 -m pip install "$(ls dist/*.whl | sort -V | tail -n 1)" --force-reinstall # install the latest build
