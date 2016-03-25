cmd = torch.CmdLine()
cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsla_Google.hdf5', '')
opt = cmd:parse(arg)

dl = require 'DataLoader'
loader = dl.create(opt.data, 8)

