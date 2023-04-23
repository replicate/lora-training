#!/bin/bash

python3 preprocessing.py $INSTANCE_DIR $INSTANCE_DIR/preprocessing --use_face_detection_instead --caption_text=$TRIGGER_WORD

accelerate config default

accelerate launch train.py