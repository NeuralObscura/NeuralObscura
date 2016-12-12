#!/usr/bin/env python

from subprocess import check_output, DEVNULL
from os import environ as env

begin_string = "==== BEGIN TEST OUTPUT ====\\n"
end_string = "==== END TEST OUTPUT ====\\n"


def test_app(device_id):
    print("iPhone must be unlocked to proceed.")
    output = check_output("""xcodebuild test \
            -project NeuralObscura.xcodeproj \
            -scheme NeuralObscura \
            -destination "platform=iOS,id={0}" \
            -only-testing:NeuralObscuraTests/NeuralStyleModelTests/testCorrectness""".format(device_id),
            stderr=DEVNULL,
            shell=True)
    output = str(output)
    start = output.find(begin_string) + len(begin_string) + len("\n")
    end = output.find(end_string)
    output = output[start:end]
    output = output.replace("\\n", "\n")
    return output

def get_device_id():
    with open(env['USER'] + ".deviceid", 'r') as f:
        return f.read().strip()

def test():
    output = test_app(get_device_id())
    print(output)

if __name__ == "__main__":
    test()
