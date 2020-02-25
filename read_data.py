import obd
import time
from obd.utils import bytes_to_int
import cv2
import numpy as np

class OBDConnection(object):

    THRTL_CODE = obd.commands.THROTTLE_POS
    SPD_CODE = obd.commands.SPEED
    RPM_CODE = obd.commands.RPM 

    def __init__(self, baud=9600, port='COM4'):
        self.obd_connection = self.establish_obd_connection(baud, port)

    def establish_obd_connection(self, baud, port, debug=False):
        if debug:
            obd.logger.setLevel(obd.logging.DEBUG)
            
        ports = obd.scan_serial()
        print(ports)

        connection = obd.OBD(port, baud) 

        while connection.query(obd.commands.SPEED).value is None:

            time.sleep(1)
            print('Retrying Connection')
            connection = obd.OBD(port, baud)

        print('Connection Established')
        return connection

    def decoder(self, messages):
        d = messages[0].data
        d = d[2:]
        v = bytes_to_int(d)/4.0
        return v

    def brute_force_scan(self, mode=1):
        hex_c = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
        valid_commands = []
        for pid1 in hex_c:
            for pid2 in hex_c:
                command = '0'+str(mode)+str(pid1)+str(pid2)
                test_com = obd.OBDCommand('Test', 'Test Command', bytes(command, encoding='utf-8'), 4, self.decoder)
                test_resp = self.obd_connection.query(test_com).value
                print(' - ' + str(test_resp))
                if test_resp is not None:
                    valid_commands.append((command, test_com, test_resp))
        return valid_commands

    def query_vehicle(self, command):
        return self.obd_connection.query(command)

if __name__ == '__main__':
    obd_connection = OBDConnection(baud=38400)
    
    '''
    while True:
        thrtl = obd_connection.query_vehicle(obd.connectionTHRTL_CODE).value
        spd = obd_connection.query_vehicle(obd_connection.SPD_CODE).value
        rpm = obd_connection.query_vehicle(obd_connection.RPM_CODE).value

        print('\rThrottle Pos: {:.2f} - Speed: {:.2f} - RPM: {:.2f} '.format(thrtl, spd, rpm), end='')
    '''
    valid_commands = obd_connection.brute_force_scan(mode=1)
    print(str(len(valid_commands))+' valid commands')
    #print(valid_commands)

    frame = np.zeros((800,800))

    place_text = [10, 50]
    while True:
        frame = np.zeros((800,800))
        place_text = [10,50]
        for com_pack in valid_commands:
            command = com_pack[1]
            command_raw = com_pack[0]
            resp = obd_connection.query_vehicle(command).value
            cv2.putText(frame, str(command_raw)+':'+str(resp), tuple(place_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            place_text[1]+=50
            if place_text[1] > frame.shape[0]-10:
                place_text[1] = 50
                place_text[0] += 200
                #print(str(resp)+' - ',end='')
        cv2.imshow('OBD2 Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print('\r',end='')