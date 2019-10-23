import obd

obd.logger.setLevel(obd.logging.DEBUG)
ports = obd.scan_serial()
print(ports)

connection = obd.OBD('COM3', baudrate=38400, protocol=1)

c = obd.commands.RPM

response = connection.query(c)

print(response.value)

connection.close()
