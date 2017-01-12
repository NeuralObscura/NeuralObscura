#!/bin/bash

DEVICE_ID=$(cat $USER.deviceid)

xcodebuild test \
  -project NeuralObscura.xcodeproj \
  -scheme NeuralObscura \
  -destination "platform=iOS,id=$DEVICE_ID" | xcpretty
