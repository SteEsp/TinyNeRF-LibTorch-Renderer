import re

def print_vec4(ws):
  vec = "vec4(" + ",".join(["{0:.3f}".format(w) for w in ws]) + ")"
  vec = re.sub(r"\b0\.", ".", vec)
  return vec

def print_mat4(ws):
  mat = "mat4(" + ",".join(["{0:.3f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
  mat = re.sub(r"\b0\.", ".", mat)
  return mat

def dump_data(tensor):
    return tensor.cpu().detach().numpy()

def serialize_to_shader(model, varname):
    linLayer = 0        
    chunks = int(model.hidden_features/4)        
        
    lin = model.layers[linLayer]
    in_w = dump_data(lin.weight)
    in_bias = dump_data(lin.bias)

    for row in range(chunks):
        line = "vec4 %s0_%d=leakyRelu(" % (varname, row)
        for ft in range(3):
            
            feature = in_w[row*4:(row+1)*4, ft]
            line += ("p.%s*" % ["x","y","z"][ft]) + print_vec4(feature) + "+"
        
        bias = in_bias[row*4:(row+1)*4]
        line += print_vec4(bias) + ");"
        print(line)
        
    #hidden layers
    linLayer = 1
    for lid in range(model.hidden_layers):
        layerID = 2 + lid * 2
        if not isinstance(model.layers[layerID], nn.Linear):
            continue
            
        layer_w = dump_data(model.layers[layerID].weight)
        layer_bias = dump_data(model.layers[layerID].bias)
        for row in range(chunks):
            line = ("vec4 %s%d_%d" % (varname, linLayer, row)) + "=leakyRelu("
            for col in range(chunks):
                mat = layer_w[row*4:(row+1)*4,col*4:(col+1)*4]
                line += print_mat4(mat) + ("*%s%d_%d"%(varname, linLayer-1, col)) + "+\n    "
            bias = layer_bias[row*4:(row+1)*4]
            
            line += print_vec4(bias)+");"
            print(line)
            
        linLayer = linLayer+1
        
    #output layer
    out_w = dump_data(model.layers[-1].weight)
    out_bias = dump_data(model.layers[-1].bias)
    line = "return "
    for row in range(chunks):
        vec = out_w[0,row*4:(row+1)*4]
        line += ("dot(%s%d_%d,"%(varname, linLayer-1, row)) + print_vec4(vec) + ")+\n    "
    print(line + "{:0.3f}".format(out_bias[0])+";")