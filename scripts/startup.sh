#!/bin/bash
STREAM_DEFAULT=ch2_bandits.py
export PYENV_VERSION=rlbook
# nohup jupyter lab &> log.jupyter&
echo streamlit run $STREAM_DEFAULT
export PYTHONPATH=~/projects/rlbook/
streamlit run streamlit_apps/$STREAM_DEFAULT
