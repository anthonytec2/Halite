#!/bin/sh
for i in $(seq 1 $1);
do
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 bots/ray-bot.py --ip=$2 --port=$3" "python3 bots/Bot2.py" --no-timeout --no-logs --no-replay &
done    
    
