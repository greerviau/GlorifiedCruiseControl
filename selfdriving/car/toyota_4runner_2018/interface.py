from __future__ import print_function, unicode_literals
from panda import Panda
import numpy as np

class Interface(object):

    def __init__(self):
        #PIDS FOR SENSORS
        self.SAS_PIN = '0x25'
        self.ACCEL_PIN = '0x2c1'
        self.BRAKE_PIN = '0x224'
        self.SPEED_PIN = '0xb4'

        #MESSAGE SLICES
        self.SAS_SLICE = slice(0,2)
        self.ACCEL_SLICE = slice(6,7)
        self.BRAKE_SLICE = slice(4,6)
        self.SPEED_SLICE = slice(5,7)

        ################################

        self.LAST_SAS = 0
        self.LAST_ACCEL = 0
        self.LAST_BRAKE = 0
        self.LAST_SPEED = 0

        self.PNDA = None
        try:
            print("Trying to connect to Panda over USB...")
            self.PNDA = Panda()

        except AssertionError:
            print("USB connection failed. Trying WiFi...")
            try:
                self.PNDA = Panda("WIFI")
            except Exception as ex:
                print(ex)
                print("WiFi connection timed out. Please make sure your Panda is connected and try again.")
                raise ex

    def parse_sas_angle_codes(self, sas_angle_hex):
        #STEERING ANGLE DECODE
        sas_angle = 0
        try:
            hex_1 = str(sas_angle_hex[1])[1]
            hex_2 = str(sas_angle_hex[2])[0]
            hex_3 = str(sas_angle_hex[2])[1]
            
            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))
            
            bin_full = bin_1+bin_2+bin_3
            
            bin_value = int(bin_full[1:], 2)
            
            if int(bin_full[0]) == 1:
                bin_value -= 2047
            
            sas_angle = bin_value*1.5
            self.LAST_SAS = sas_angle
        
        except Exception as ex:
            sas_angle = self.LAST_SAS
            pass

        return sas_angle

    def parse_accel_codes(self, accel_hex_codes):
        #GAS PEDAL DECODE
        accel_pos = 0
        try:
            hex_1 = str(accel_hex_codes[1])[0]
            hex_2 = str(accel_hex_codes[1])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))

            bin_full = bin_1+bin_2

            bin_value = int(bin_full, 2)

            accel_pos = bin_value * 0.005
            self.LAST_ACCEL = accel_pos

        except Exception as ex:
            accel_pos = self.LAST_ACCEL
            pass

        return accel_pos

    def parse_brake_codes(self, brake_hex_codes):
        #BRAKE PEDAL DECODE
        brake_pos = 0
        try:
            hex_1 = str(brake_hex_codes[1])[1]
            hex_2 = str(brake_hex_codes[2])[0]
            hex_3 = str(brake_hex_codes[2])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))

            bin_full = bin_1+bin_2+bin_3

            bin_value = int(bin_full[1:], 2)

            brake_pos = bin_value / 2047.0
            self.LAST_BRAKE = brake_pos

        except Exception as ex:
            brake_pos = self.LAST_BRAKE
            pass

        return brake_pos

    def parse_speed_codes(self, speed_hex_codes):
        #SPEED DECODE
        speed = 0
        try:
            hex_1 = str(speed_hex_codes[1])[0]
            hex_2 = str(speed_hex_codes[1])[1]
            hex_3 = str(speed_hex_codes[2])[0]
            hex_4 = str(speed_hex_codes[2])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))
            bin_4 = "{0:04b}".format(int(hex_4, 16))

            bin_full = bin_1+bin_2+bin_3+bin_4

            bin_value = int(bin_full, 2)

            speed = bin_value * 0.01
            self.LAST_SPEED = speed

        except Exception as ex:
            speed = self.LAST_SPEED
            pass
        
        return speed

    def get_can_messages(self):
        can_recv = self.PNDA.can_recv()
        
        sas_angle_hex_codes = []
        accel_hex_codes = []
        brake_hex_codes = []
        speed_hex_codes = []
        for address, _, dat, src  in can_recv:
            
            #CONVERT DATA TO AN ARRAY OF HEX MESSAGES
            dat_array = ["\\x%02x" % i for i in dat]
            dat_hex = "".join("\\x%02x" % i for i in dat)

            #EXTRACT MESSAGES
            
            if hex(address) == self.SAS_PIN:
                print(src)
                sas_angle_hex_codes = dat_array[self.SAS_SLICE]
            
            elif hex(address) == self.ACCEL_PIN:
                accel_hex_codes = dat_array[self.ACCEL_SLICE]

            elif hex(address) == self.BRAKE_PIN:
                brake_hex_codes = dat_array[self.BRAKE_SLICE]

            elif hex(address) == self.SPEED_PIN:
                speed_hex_codes = dat_array[self.SPEED_SLICE]

        #SPLIT INTO INDIVIDUAL HEX MESSAGES
        sas_angle_hex_codes = "".join(sas_angle_hex_codes).split('\\x')
        accel_hex_codes = "".join(accel_hex_codes).split('\\x')
        brake_hex_codes = "".join(brake_hex_codes).split('\\x')
        speed_hex_codes = "".join(speed_hex_codes).split('\\x')
        
        #DECODING HEX MESSAGES
        #MUST BE ADJUSTED FOR DIFFERENT VEHICLES
        
        sas_angle = self.parse_sas_angle_codes(sas_angle_hex_codes)
        accel_pos = self.parse_accel_codes(accel_hex_codes)
        brake_pos = self.parse_brake_codes(brake_hex_codes)
        speed = self.parse_speed_codes(speed_hex_codes)

        speed *= 0.621371 #kph to mph conversion
        
        #FORMAT DATA
        sas_angle = float('{0:.4f}'.format(sas_angle))
        accel_pos = float('{0:.4f}'.format(accel_pos))
        brake_pos = float('{0:.4f}'.format(brake_pos))
        speed = float('{0:.4f}'.format(speed))

        return sas_angle, accel_pos, brake_pos, speed

    def send_can_messages(self, messages):
        #not sure if this works yet
        for address, message in messages:
            self.PNDA.can_send(address, message, 0)
