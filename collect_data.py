from road_vision import VPS
import sys, csv
import cv2

file_name = sys.argv[1]

obd = OBDConnection()
vps = VPS(show_visuals=False, position_camera=True, invert=True, readout=False, record_file = file_name, record_raw=True, record_processed=True, return_data=True)

#cap = cv2.VideoCapture('tests/test_10/test_10_raw.mp4')
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

with open('tests/'+file_name+'/'+file_name+'.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['left_curve', 'right_curve', 'lane_curve', 'vehicle_offset', 'turn', 
                    'v1_conf', 'v1_type', 'v1_size', 'v1_lane', 'v1_x', 'v1_y',
                    'v2_conf', 'v2_type', 'v2_size', 'v2_lane', 'v2_x', 'v2_y',
                    'v3_conf', 'v3_type', 'v3_size', 'v3_lane', 'v3_x', 'v3_y',
                    'v4_conf', 'v4_type', 'v4_size', 'v4_lane', 'v4_x', 'v4_y',
                    'v5_conf', 'v5_type', 'v5_size', 'v5_lane', 'v5_x', 'v5_y',
                    'spd'])
    while True:
        ret, frame = cap.read()
        spd = obd.query_vehicle(obd.SPD_CODE).value

        if not ret:
            break
        lane_data, vehicle_data, frame = vps.road_vision(frame)
        #print(lane_data)
        #print(vehicle_data)

        vector = []
        vector.extend(list(lane_data))
        for vehicle in vehicle_data:
            vector.extend(list(vehicle))

        vector.append(spd)
        #print(vector)
        writer.writerow(vector)

        cv2.imshow('VPS', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()