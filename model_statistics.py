def model_statistics(model, dataset):
    
    if dataset == 'CIFAR-10':

        if model == 'AlexNet':
            layer_flops = [0, 2e6, 2e6, 2e6, 2e6, 2e6]
            inter_data_size = [12, 64, 128, 64, 64, 0.3] # KB

        elif model == 'VGG-16':
            layer_flops = [0, 2172160, 37945344, 18833408, 37847040,
                            18989056, 37797888, 37797888, 18833408, 37773312,
                            37773312, 21247488, 21247488, 24336640]
            inter_data_size = [12, 256, 64, 128, 32,
                                64, 64, 16, 32, 32,
                                18, 18, 18, 0.35]
            
        elif model == 'SqueezeNet-1.0':
            layer_flops = [0, 2617088, 140352, 2261504, 175168,
                            2261504, 612480, 8045568, 279680,
                            1598464, 411328, 3429888, 493248,
                            3503616, 821504, 5851136, 492544,
                            2662400, 15274]
            inter_data_size = [12, 74, 12, 98, 12,
                                98, 24, 36, 5,
                                36, 7, 54, 7,
                                54, 9, 32, 4,
                                32, 0.35]
            
        elif model == 'Inception-V3':
            layer_flops = [0, 1213696, 8290560, 16798208, 4213248,
                            93562368, 147474432, 159860736, 164247552,
                            173005248, 156986368, 204611968, 204611968,
                            259176192, 120527360, 126213504, 151677056]
            inter_data_size = [12, 128, 113, 196, 245,
                                432, 576, 648, 648,
                                363, 363, 363, 363,
                                363, 125, 200, 0.3]
            
        elif model == 'ResNet-34':
            layer_flops = [0, 2516224, 4734976, 4734976, 4734976, 3674112,
                            4726784, 4726784, 4726784, 3643392, 4722688,
                            4722688, 4722688, 4722688, 4722688, 14725120,
                            18882560, 18821376]
            inter_data_size = [12, 16, 16, 16, 16,
                                8, 8, 8, 8, 4,
                                4, 4, 4, 4, 4,
                                8, 8, 0]
            
    elif dataset == 'ImageNet':

        if model == 'Inception-V3':
            layer_flops = [0, 19568544, 110255680, 220970112, 15036032,
                            373102592, 162037152, 174468576, 178220000,
                            198543680, 186826752, 243505152, 243505152,
                            308441088, 143344640, 119823744, 135504341]
            inter_data_size = [588, 1540, 1485, 729, 911,
                                469, 625, 703, 703,
                                432, 432, 432, 432,
                                432, 125, 200, 0.3]
            
        elif model == 'ResNet-34':
            layer_flops = [0, 128212992, 232013824, 232013824, 232013824,
                            181481472, 231612416, 231612416, 231612416,
                            179083264, 231411712, 231411712, 231411712,
                            231411712, 231411712, 720973824, 925245440, 925245440]
            inter_data_size = [588, 784, 784, 784, 784,
                                392, 392, 392, 392, 196,
                                196, 196, 196, 196, 196,
                                392, 392, 0.4]

    elif dataset == 'MNIST':

        if model == 'LeNet':
            layer_flops = [0, 123900, 241600, 48000, 10080, 840]
            inter_data_size = [4, 1.56, 6.25, 0.47, 0.33, 0.04] # KB

    return layer_flops, inter_data_size


def get_split_model_statistics(model, dataset, split_layer: int, backprop_compute_coef = 2.0):
    
    layer_flops, inter_data_size = model_statistics(model=model, dataset=dataset)
    assert len(layer_flops) == len(inter_data_size)

    if split_layer < 1 or split_layer > len(layer_flops): # 1 to N
        raise ValueError("Invalid split layer!")
    
    flops_f_device = sum(layer_flops[:split_layer])
    flops_f_server = sum(layer_flops[split_layer:])
    assert flops_f_device + flops_f_server == sum(layer_flops)

    flops_b_device = flops_f_device * backprop_compute_coef
    flops_b_server = flops_f_server * backprop_compute_coef

    inter_data_trans = inter_data_size[split_layer - 1]
    
    return flops_f_device, flops_f_server, flops_b_device, flops_b_server, inter_data_trans
