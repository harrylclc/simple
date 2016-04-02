local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Sequence to Sequence Training')
    cmd:text()
    cmd:text('Options:')
    -- General
    cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsela.hdf5', 'training data and word2vec data')
    cmd:option('-vocab', '/data_giles/cul226/simple/preprocessed/newsela.vocab', 'vocabulary file')
    cmd:option('-gpuid', 1, 'gpu id')
    cmd:option('-threads', 4 , 'number of threads')
    -- Training
    cmd:option('-epochs', 1, 'number of epochs')
    cmd:option('-batchSize', 8, 'batch size')
    cmd:option('-hiddenSize', 100, 'rnn output size')
    cmd:option('-useBN', false, 'use batch normalization or not')
    cmd:option('-log', false, 'optim logging')
    -- save/load checkpoint
    cmd:option('-save_every', 3000, 'save model every # iterations')
    cmd:option('-init_from', '', 'init from checkpoint model')
    cmd:option('-checkpoint_dir', '/data_giles/cul226/simple/models', 'checkpoint dir')
    cmd:option('-savefile','enc-dec','save to checkpoint_dir/{savefile}')
    -- Debug/print info
    cmd:option('-print_every', 5, 'print every # iterations')
    cmd:option('-print_sent_every', 100, 'print model predictions every # iters')
    -- Optimization
    cmd:option('-learningRate', 0.001, 'learning rate')
    cmd:option('-momentum', 0.9, 'momentum')
    cmd:option('-opt', 'sgd', 'which optimization method to use')
    cmd:option('-gradClip', 10, 'clip gradients at this value')

    local opt = cmd:parse(arg or {})
    return opt
end

return M
