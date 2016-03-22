require 'nn'
require 'rnn'

local EncoderDecoder = {}
EncoderDecoder.__index = EncoderDecoder

function EncoderDecoder.create(w2v)
    local self = {}
    setmetatable(self, EncoderDecoder)

    local vocabSize = w2v:size(1)
    local vecSize = w2v:size(2)
    -- pre-trained w2v
    local lookup = nn.LookupTable(vocabSize, vecSize)
    lookup.weight:copy(w2v)
    lookup.weight[1]:zero()
    lookup.accGradParameters = nil

    -- encoder
    local enc = nn.Sequential()
    enc:add(lookup)
    enc:add(nn.SplitTable(1, 2))
    local encLSTM = nn.LSTM(vecSize, vecSize)
    enc:add(nn.Sequencer(encLSTM))
    enc:add(nn.SelectTable(-1))

    -- decoder
    local dec = nn.Sequential()
    dec:add(lookup)
    dec:add(nn.SplitTable(1, 2))
    local decLSTM = nn.LSTM(vecSize, vecSize)
    dec:add(nn.Sequencer(decLSTM))
    dec:add(nn.Sequencer(nn.Linear(vecSize, vocabSize)))
    dec:add(nn.Sequencer(nn.LogSoftMax()))

    self.model = nn.Container():add(enc):add(dec)
    self.enc = enc
    self.dec = dec
    self.encLSTM = encLSTM
    self.decLSTM = decLSTM

    return self
end

function EncoderDecoder:forward(encInSeq, decInSeq)
    local encOut = self.enc:forward(encInSeq)
    EncoderDecoder.forwardConnect(self.encLSTM, self.decLSTM, encInSeq:size(2))
    local decOut = self.dec:forward(decInSeq)
    return decOut
end

function EncoderDecoder:backward(encInSeq, decInSeq, gradOutput)
    local zeroTensor = torch.Tensor(encInSeq:size(1), self.encLSTM.outputSize)
    self.dec:backward(decInSeq, gradOutput)
    EncoderDecoder.backwardConnect(self.encLSTM, self.decLSTM)
    self.enc:backward(encInSeq, self.zeroTensor)
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function EncoderDecoder.forwardConnect(encLSTM, decLSTM, seqLen)
   decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[seqLen])
   decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[seqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function EncoderDecoder.backwardConnect(encLSTM, decLSTM)
   encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
   encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

return EncoderDecoder
