require 'rnn'

cmd = torch.CmdLine()

cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-model', '', 'checkpoint model')

opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
end

checkpoint = torch.load(opt.model)

checkpoint.encoderDecoder.model:double()

local saveFile = opt.model .. '_cpu.t7'
print('save to' .. saveFile)
torch.save(saveFile, checkpoint)
