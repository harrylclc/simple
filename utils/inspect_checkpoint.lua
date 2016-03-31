require 'gnuplot'
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

model = torch.load(opt.model)

local trainLosses = model.trainLosses
losses = torch.Tensor(#trainLosses)
for i = 1, #trainLosses do
    losses[i] = trainLosses[i]
end

-- plot loss

gnuplot.pdffigure(opt.model ..  '_loss.pdf')
gnuplot.plot(losses)
gnuplot.xlabel('# iter')
gnuplot.ylabel('err')
gnuplot.plotflush()

