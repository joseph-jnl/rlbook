#!/bin/bash
STREAM_DEFAULT=ch2_bandits.py
pyenv activate rlbook
nohup jupyter lab &> log.jupyter&
echo streamlit run $STREAM_DEFAULT
streamlit run streamlit/$STREAM_DEFAULT
