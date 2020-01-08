def parse_net_config(net_config):
    if isinstance(net_config, list):
        return net_config
    elif isinstance(net_config, str):
        str_configs = net_config.split('|')
        return [eval(str_config) for str_config in str_configs]
    else:
        raise TypeError

def load_net_config(path):
    with open(path, 'r') as f:
        net_config = ''
        while True:
            line = f.readline().strip()
            if line:
                net_config += line
            else:
                break
    return net_config
