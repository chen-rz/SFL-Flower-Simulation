def model_statistics(model):

    if model == 'alexnet':
        layer_flops = [70857600, 224368704, 112270080, 149606912, 99810048, 54592522]
        inter_data_size = [1492992, 1038336, 2076672, 1384448, 294912, 320] # bit

    elif model == 'vgg11':
        layer_flops = [96337920, 929660928, 926449664, 1852096512, 925646848, 1850892288, 462622720, 462723072, 119620106]
        inter_data_size = [25690112, 12845056, 25690112, 6422528, 12845056, 3211264, 3211264, 802816, 320]

    else:
        raise ValueError("Invalid model")

    assert len(layer_flops) == len(inter_data_size)
    
    return len(layer_flops), layer_flops, inter_data_size


def get_split_model_statistics(model, split_layer: int, backprop_compute_coef = 2.0):
    
    layer_num, layer_flops, inter_data_size = model_statistics(model=model)

    if split_layer < 1 or split_layer > layer_num: # 1 to N
        raise ValueError("Invalid split layer")
    
    flops_f_device = sum(layer_flops[:split_layer])
    flops_f_server = sum(layer_flops[split_layer:])
    assert flops_f_device + flops_f_server == sum(layer_flops)

    flops_b_device = flops_f_device * backprop_compute_coef
    flops_b_server = flops_f_server * backprop_compute_coef

    inter_data_trans = inter_data_size[split_layer - 1]
    
    return flops_f_device, flops_f_server, flops_b_device, flops_b_server, inter_data_trans


FLP_F_D = 0
FLP_F_S = 1
FLP_B_D = 2
FLP_B_S = 3
INT_TRANS = 4
