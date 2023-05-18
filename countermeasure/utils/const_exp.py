num_classes = 95
# the cam score threshold of informative regions
threshold = 55

# TAM parameters
max_matrix_len = 1800
maximum_load_time = 80.
time_slot = 80. / (max_matrix_len - 1)

'''
    Our traffic morphing strategy uses the following parameters:
    delta_up, delta_down: Boundaries parameters for the number of packets sent in a time slot
    fill_num_up, fill_num_down: When there is no packet in a time slot, we fill it with a constant number of packets
    U: Threshold for the number of delayed packets sending in a time slot
    N: Number of selected informative regions of label c'
    D: Threshold for the total number of delayed packets
    
'''

delta_up = 0.2
delta_down = 0.3

fill_num_up = 0
fill_num_down = 1

U = 15
N = 15
D = 400

save = True
# You can change the suffix to 'train' or 'test' to generate the defended traces separately
suffix = 'train'
total_trace = 85500 if suffix == 'train' else 9500

# TODO: the defended trace will be save into output_path, please change it to your own path
output_path = ''