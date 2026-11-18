#!/bin/bash

wget -O TartanAir.pickle "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/ADRnGzAu9vnvU066D8SkeNc/TartanAir.pickle?rlkey=4j336vcc4h324j4p4ji3ppl0t" && mv TartanAir.pickle datasets/
wget -O models.zip "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t" && unzip models.zip
wget -O movies.zip "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AHv2iLIeoYDKlkQKZEvnovc/movies.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t" && unzip movies.zip
