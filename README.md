# tf_workspace
Originally based on UIUC and IBM Lab, see: https://github.com/yscacaca/DeepSense

Tensorflow workspace for indicator learning project

**sensor_data:**

**raw_data:**

Raw .mat data directly collected from PreScan in the same scenario. For detailed explaination, please refer to the sensor section in PreScan manual: https://virginia.app.box.com/s/gzb8r2dwc78r4336ntpccm0f1m5qw05r


lidar_truth.mat: noise-free lidar information, horizontal scan for 120 degrees. The lidar shoots 60 beams at a time and record the information of the returned beam. Useful fields: Range_m (distance to object detected), Theta_m (bearing angle to object detected), Target_ID (ID of object detected). The information of the same object is stored at the same place of different fields.

lidar_1.mat: same lidar but with 5 degrees of noise

lidar_2.mat: same lidar but with 2.5 degrees of noise

radar_truth.mat: noise-free radar information, horizontal scan for 60 degrees. It has the same struct as lidar, but shoots 7 beams at a time instead.

radar_1.mat: same radar but with 2 degrees of noise

radar_2.mat: same radar but with 1 degree of noise

curr_state.mat: CAN bus output of current speed and bearing angle, can be used for control training in the future

lanes.mat: information of the lanes, including SliceCount (number of lanes) and other information, currently not used.


For training purpose, the information collected by noisy sensors, specifically distance and bearing angle to the closest 4 objects measured will be trained to learn the ground truths.
