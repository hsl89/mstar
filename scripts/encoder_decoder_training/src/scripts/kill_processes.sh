#!/bin/bash

#useful to kill all running training processes during development
#otherwise pytorch lightning shutdown takes a long time using SIGTERM
#use at your own risk, kills all python3 string matches

kill -9 $(ps -e | grep python3 | awk {'print $1'})
