"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import serial
import time
import pandas as pd
import signal

CONTINUE = True


def handle_exit_signal(signal, frame):
    print("sigint")
    global CONTINUE
    CONTINUE = False


def write_log(log, timestamps, output_name):
    column_headers = [
        "time (ms)",
        "input voltage (mV)",
        "input current (mA)",
        "input power (mW)",
        "on/off",
        "channel 0 voltage (mV)",
        "channel 0 current (mA)",
        "channel 0 power (mW)",
        "channel 0 on/off",
        "channel 0 interrupts",
        "channel 1 voltage (mV)",
        "channel 1 current (mA)",
        "channel 1 power (mW)",
        "channel 1 on/off",
        "channel 1 interrupts",
        "checksum A",
        "checksum B",
    ]

    log_split = [x.split(",") for x in log]

    df = pd.DataFrame(log_split, columns=column_headers)
    df["timestamp"] = timestamps

    df.to_csv(f"{output_name}", index=False)


def main(args):
    # set up a way to exit the while loop
    signal.signal(signal.SIGINT, handle_exit_signal)

    power_supply = serial.Serial(args.port, args.baud, timeout=1)

    log = []
    timestamps = []
    # clear anything in the input buffer before beginning reading
    power_supply.reset_input_buffer()
    while len(power_supply.readline()) != 81:
        power_supply.readline()

    print("ready to begin logging")
    while CONTINUE:
        line = power_supply.readline()
        # log the time we finished reading the line, indicating the
        # host time of the measurement (with some delay)
        timestamps.append(time.time_ns())
        if len(line) != 81:
            raise IOError()
        log.append(line)

    with open("temp.log", "w") as file:
        file.writelines([x.decode("utf-8") for x in log])

    log = [x.decode("utf-8").replace("\r\n", "") for x in log]

    write_log(log, timestamps, args.output_name)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("output_name", help="Name for output file, without extension.")

    # Optional argument flag which defaults to False
    parser.add_argument(
        "-p",
        "--port",
        help="Interface to connect to device, i.e. /dev/cu.usbserial-1130",
        default="/dev/cu.usbserial-1130",
    )
    parser.add_argument(
        "-b", "--baud", help="Baud rate to connect to the device", default=921600
    )

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity (-v, -vv, etc)"
    )

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__),
    )

    args = parser.parse_args()
    main(args)
