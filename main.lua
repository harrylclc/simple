require 'hdf5'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsla_Google.hdf5', 'training data and word2vec data')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-save_every', 3000, 'save model every # iterations')
cmd:option('-checkpoint_dir', '/data_giles/cul226/simple/models', 'checkpoint dir')
cmd:option('-savefile','enc-dec','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-print_every', 1, 'print every # iterations')
cmd:option('-epochs', 1, 'number of epochs')
cmd:option('-batchSize', 8, 'batch size')
cmd:option('-learningRate', 0.001, 'learning rate')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-threads', 4 , 'number of threads')
cmd:option('-init_from', '', 'init from checkpoint model')
cmd:option('-opt', 'sgd', 'which optimization method to use')

-- options
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
else
    torch.setnumthreads(opt.threads)
    print('<torch> set # threads to ' .. torch.getnumthreads())
end

local DL = require 'DataLoader'
local loader = DL.create(opt.data, opt.batchSize)

local EncDec = require 'EncoderDecoder'
local encoderDecoder

if string.len(opt.init_from) > 0 then
    print('init model from', opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    encoderDecoder = checkpoint.encoderDecoder
    setmetatable(encoderDecoder, EncDec)
else
    encoderDecoder = EncDec.create(loader.w2v)
end


local model = encoderDecoder.model
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local toTable = nn.SplitTable(1, 1)

if opt.gpuid >= 0 then
    model:cuda()
    criterion:cuda()
    toTable:cuda()
end

local optimMethod, optimState

if opt.opt == 'sgd' then
    optimState = {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
    }
    optimMethod = optim.sgd
elseif opt.opt == 'rmsprop' then
    optimState = {
        learningRate = opt.learning_rate,
        alpha = 0.95
    }
    optimMethod = optim.rmsprop
else
    error('unknown optimization method')
end

params, gradParams = model:getParameters()

local iterations = opt.epochs * loader.numBatches
local iterationsPerEpoch = loader.numBatches

trainLosses = {}
for i = 1, iterations do
    model:training()
    local epoch = i / loader.numBatches
    local timer = torch.Timer()

    local encInSeq, decInSeq, decOutSeq = loader:nextBatch()
    if opt.gpuid >= 0 then
        encInSeq = encInSeq:cuda()
        decInSeq = decInSeq:cuda()
        decOutSeq = decOutSeq:cuda()
    end
    decOutSeq = toTable:forward(decOutSeq)

    local feval = function(x)
        if x ~= params then
            params:copy(x)
        end

        gradParams:zero()
        -- forward
        local decOut = encoderDecoder:forward(encInSeq, decInSeq)

        local err = criterion:forward(decOut, decOutSeq)

        -- backward
        local gradOutput = criterion:backward(decOut, decOutSeq)
        encoderDecoder:backward(encInSeq, decInSeq, gradOutput)

        return err, gradParams
    end

    local _, loss = optimMethod(feval, params, optimState)

    local time = timer:time().real
    local trainLoss = loss[1]
    trainLosses[i] = trainLoss

    if i % opt.save_every == 0 or i % loader.numBatches == 0 or i == iterations then
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, trainLoss)
        print('iter ', i, 'save checkpoint', savefile)
        local checkpoint = {}
        checkpoint.encoderDecoder = encoderDecoder
        checkpoint.opt = opt
        checkpoint.trainLosses = trainLosses
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
        print('done')
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, trainLoss, gradParams:norm() / params:norm(), time))
    end

    if i % 10 == 0 then
        collectgarbage()
    end

end

