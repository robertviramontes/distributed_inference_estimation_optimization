"""
Server for listening to requests to start/stop power logging, with some metadata
"""

__author__ = "Robert Viramontes"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import socket
from multiprocessing import Process, Queue
from smartpower3_logger import write_log
import serial
import time


def logging(output_name, port, baud, run_q, log_q):
    """Main logging function.
        output_name: Name of the file to save, including .csv extension
        port: serial port the SMARTPOWER3 is connected to
        baud: baud rate for serial connection to SMARTPOWER3
        loq_q: message queue to communicate with main thread
    """
    log = []
    errors = []
    timestamps = []

    with serial.Serial(port, baud, timeout=1, dsrdtr=True) as power_supply:
        # clear anything in the input buffer before beginning reading
        power_supply.reset_input_buffer()
        while len(power_supply.readline()) != 81:
            power_supply.readline()

        print("ready to begin logging")
        while not log_q.empty():
            line = power_supply.readline()
            # log the time we finished reading the line, indicating the
            # host time of the measurement (with some delay)
            timestamps.append(time.time_ns())
            if len(line) != 81:
                errors.append((line, time.time_ns()))
            else:
                log.append(line)
        
        while not run_q.empty():
            print("device still saving")
            power_supply.readline()

    with open("temp.log", "w") as file:
        file.writelines([x.decode("utf-8") for x in log])
    with open(f"{output_name}.errors", "w") as errors_file:
        errors_file.writelines(
            [(entry[0].decode("utf-8"), entry[1]) for entry in errors]
        )

    log = [x.decode("utf-8").replace("\r\n", "") for x in log]

    write_log(log, timestamps, output_name)


def main(args):
    """Main entry point of the app"""
    # based on the echo server example from Python docs
    # https://docs.python.org/3/library/socket.html#example
    logging_process = None
    run_q = Queue(1)
    log_q = Queue(1)

    # Start a server and listen for communications from the DUT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.server_address, args.server_port))
        s.listen(1)
        
        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                print(data.decode("utf-8"))
                commands = data.decode("utf-8").split(",")
                # expect a tuple with the
                if "start" in commands and len(commands) == 2:
                    # spawn logging process and keep running on non-empty queue
                    run_q.put("run")
                    log_q.put("start_logging")
                    logging_process = Process(
                        target=logging, args=(commands[1], args.port, args.baud, run_q, log_q)
                    )
                    logging_process.start()

                    # ack the start message
                    conn.sendall(b"started")
                elif "stop" in commands:
                    # stop logging process by emptying the queue
                    log_q.get()
                    
                    # ack the stop message
                    conn.sendall(b"stopped")
                elif "shutdown" in commands:
                    print(run_q.empty())
                    run_q.get()
                    logging_process.join()
                    conn.sendall(b"ack_shutdown")
                else:
                    print(commands)
                    conn.sendall(b"bad message")


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("server_address", help="Required positional argument")
    parser.add_argument("server_port", help="Port to listen on", type=int)

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
