#!/usr/bin/env bash
set -Eeuo pipefail
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for i in $(seq 1 15); do
  if ! [ -s "output-DE-F$i.txt" ]; then
    sed -i "s/ F[[:digit:]]\+ / F$i /g" main.cpp
    sed -i "s/output-DE-F[[:digit:]]\+.txt/output-DE-F$i.txt/g" main.cpp
    make
    ./main
  else
    echo "output-DE-F$i.txt";
  fi
done
