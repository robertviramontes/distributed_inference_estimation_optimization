# Power Logging

Power log scripts are meant to be run on a host PC that is connected to both the ODROID SmartPower 3 and device under test (DUT), being powered by the SmartPower 3. 
Tested on Windows PC. Should work on macOS or Linux-based PCs, with no or minimal modifications. 

Note that this is specific to this repository, and expects a particular pattern of communication with the DUT to work. A more general data logger for smart power can be found at the [Hardkernel GitHub](https://github.com/hardkernel/smartpower3/tree/master/contrib).

## Usage example

```
python3 powerlog_server.py 192.168.137.1 8010 -p COM3 -b 500000
```

Starts a server on the host PC that is listening at `192.168.137.1:8010` for messages from the DUT. It will log from the SmartPower 3 which is at serial port `COM3` and configured for a baud rate of `500_000`. 

A sample setup with Jetson Nano as DUT, Windows Laptop as host, and SmartPower 3.
![Image of lab setup](power_setup_labeled.png)

> [!NOTE]
> Notice that the filename for saving is not included as an arg! This is because the DUT will send the filename when requesting logging to start. This allows multiple tests to run on the DUT without restarting the logging server.
> 


### Combining timing and power profiles into joint timing-energy profile
