def generate_bbox(input_np, ratios):
    assert len(input_np) == len(ratios)

    bbox = []
    for im, ratio in zip(input_np, ratios):
        height, width = im.shape[:2]
        xmin = int(float(ratio[0]) / 20 * width)
        ymin = int(float(ratio[1]) / 20 * height)
        xmax = int(float(ratio[2]) / 20 * width)
        ymax = int(float(ratio[3]) / 20 * height)


        bbox.append((xmin, ymin, xmax, ymax))

    return bbox


def crop_input(input_np, bbox):
    assert len(input_np) == len(bbox)

    result = [transform.resize(im[ymin:ymax, xmin:xmax], (227, 227), mode = 'constant')
            for im, (xmin, ymin, xmax, ymax) in zip(input_np, bbox)]

    return np.asarray(result, dtype = np.float(32))



def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding = "VALID", group = 1):
    
    # Taker from https://github.com/etheron/caffe-tensorflow
    convolve = lambda i, k: tf.nn.conv2s(i, k, [1, s_h, s_w, 1], padding = padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf-split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)

    return tf.nn.bias_add(conv, biases)
        


