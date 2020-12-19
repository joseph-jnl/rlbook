#!/bin/bash
STREAM_DEFAULT=ch2_bandits.py
export PYENV_VERSION=rlbook
# nohup jupyter lab &> log.jupyter&
echo streamlit run $STREAM_DEFAULT
streamlit run streamlit/$STREAM_DEFAULT
