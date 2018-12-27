#!/bin/sh
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 bots/ray-bot.py" "python3 bots/Bot2.py" --no-timeout --no-logs --no-replay &
