------------------------------------------------------------------------
-- Based on GRU
--[[ GRU ]]--
-- Gated Recurrent Units architecture.
-- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-gruGRU-rnn-with-python-and-theano/
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
require 'nn'
require 'nngraph'
require 'dpnn'
require 'rnn'
require '../utils/Linear3D'
require '../utils/ConvertTable'

local GRUAttention, parent = torch.class('nn.GRUAttention', 'nn.AbstractRecurrent')

function GRUAttention:__init(inputSize, outputSize, nFeaturesDim, predSize, rho)

   parent.__init(self, rho or 9999)
   self.inputSize = inputSize
   self.outputSize = outputSize   
   self.nFeaturesDim = nFeaturesDim
   self.predSize = predSize
   -- build the model
   self.cell = self:buildCell()
   self.recurrentModule = self:buildModel(0.5)
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
   
   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor() 
   
   self.cells = {}
   self.gradCells = {}

   self.features = torch.Tensor()
   self.gradFeaturesAcc = torch.Tensor()
end

-------------------------- factory methods -----------------------------
function GRUAttention:buildCell()
   -- input : {input, prevOutput}
   -- output : {output}
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize+self.nFeaturesDim, 2*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)

   local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
   local gates = nn.Sequential()
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   gates:add(nn.Reshape(2,self.outputSize))
   gates:add(nn.SplitTable(1,2))
   local transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable()
   concat:add(nn.Identity()):add(gates)
   local seq = nn.Sequential()
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- x(t), s(t-1), r, z

   -- Rearrange to x(t), s(t-1), r, z, s(t-1)
   local concat = nn.ConcatTable()  -- 
   concat:add(nn.NarrowTable(1,4)):add(nn.SelectTable(2))
   seq:add(concat):add(nn.FlattenTable())

   -- h
   local hidden = nn.Sequential()
   local concat = nn.ConcatTable()
   local t1 = nn.Sequential()
   t1:add(nn.SelectTable(1)):add(nn.Linear(self.inputSize+self.nFeaturesDim, self.outputSize))
   local t2 = nn.Sequential()
   t2:add(nn.NarrowTable(2,2)):add(nn.CMulTable()):add(nn.LinearNoBias(self.outputSize, self.outputSize))
   concat:add(t1):add(t2)
   hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())
   
   local z1 = nn.Sequential()
   z1:add(nn.SelectTable(4))
   z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

   local z2 = nn.Sequential()
   z2:add(nn.NarrowTable(4,2))
   z2:add(nn.CMulTable())

   local o1 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(z1)
   o1:add(concat):add(nn.CMulTable())

   local o2 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(o1):add(z2)
   o2:add(concat):add(nn.CAddTable())

   seq:add(o2)
   
   return seq
end

function GRUAttention:buildModel(rate)
   -- recurrentModule input: {input, prevOutput, features} 
   --       input: (nb_batch) * inputSize
   --       prevOutput: (nb_batch) * outputSize
   --       features: (nb_batch) * T * nFeaturesDim
   -- recurrentModule output: {output}
   --       output: (nb_batch) * outputSize

   -- cell input : {input, prevOutput}
   --       input: (nb_batch) * (inputSize+nFeaturesDim)
   --       prevOutput: (nb_batch) * outputSize
   -- cell output : {output}
   --       output: (nb_batch) * outputSize

   local input = nn.Identity()() -- y_{i-1}, nb_batch * intputSize
   local features = nn.Identity()() -- nb_batch * T * nFeaturesDim
   local prevOutput = nn.Identity()() -- nb_batch * outputSize

   local features_fc = nn.Linear3D(self.nFeaturesDim, self.nFeaturesDim*2)(features)
   local features_split = nn.SplitTable(1, 2)(features_fc)
   local prevOutput_fc = nn.Linear(self.outputSize, self.nFeaturesDim*2, false)(prevOutput)

   local sum = nn.CAddTensorTable()({prevOutput_fc, features_split})
   local sum_tilt = nn.Linear3D(self.nFeaturesDim*2, 1)(nn.Tanh()(nn.ConvertTable()(sum)))

   local attention = nn.Transpose({1, 2})(nn.View(-1):setNumInputDims(2)(sum_tilt))
   local attention_norm = nn.View(-1, 1):setNumInputDims(1)(nn.SoftMax()(attention))
   local context = nn.View(-1):setNumInputDims(2)(nn.MM(true, false)({features, attention_norm}))

   local concat = nn.JoinTable(1, 1)({context, input})
   local output = self.cell({concat, prevOutput}) -- hidden: named as output to follow convension

   local h_tilt = nn.JoinTable(1, 1)({concat, output})
   local h_tilt_fc = nn.Linear(self.nFeaturesDim+self.inputSize+self.outputSize, self.predSize)(nn.Dropout(rate)(h_tilt))
   local pred = nn.LogSoftMax()(h_tilt_fc)

   local model = nn.gModule({input, prevOutput, features}, {output, attention_norm, pred})
   return model
end

function GRUAttention:setFeatures(features)
   -- features: nb_batch * T * nb_dim
   assert(torch.isTensor(features), "features should be 2-d or 3-d tensor: (nb_batch *) T * nb_dim")
   assert((features:nDimension() == 2 or features:nDimension() == 3), "featuresshould be 2-d or 3-d tensor: (nb_batch *) T * nb_dim")
   assert(features:size(features:nDimension()) == self.nFeaturesDim, "features dimension not match nFeaturesDim")

   self.features = self.features:resizeAs(features):copy(features)
   self.gradFeaturesAcc = self.gradFeaturesAcc:resizeAs(features):zero()
end

------------------------- forward backward -----------------------------
function GRUAttention:updateOutput(input)
   local prevOutput
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      if input:dim() == 2 then
         self.zeroTensor:resize(input:size(1), self.outputSize):zero()
      else
         self.zeroTensor:resize(self.outputSize):zero()
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[self.step-1]
   end

   -- output(t) = nngraph{input(t), output(t-1), self.features}
   assert(self.features:nDimension() > 0)
   local output, attention_norm, pred
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, attention_norm, pred = unpack(recurrentModule:updateOutput{input, prevOutput, self.features})
   else
      output, attention_norm, pred = unpack(self.recurrentModule:updateOutput{input, prevOutput, self.features})
   end
   
   self.outputs[self.step] = output
   self.output = output
   self.attention_norm = attention_norm
   self.pred = pred
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradFeaturesAcc:zero()
   -- note that we don't return the cell, just the output
   return self.pred
end

function GRUAttention:_updateGradInput(input, gradOutputTable)
   assert(type(gradOutputTable) == 'table', 'gradOutput should be table')
   local gradOutput = gradOutputTable[1]

   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   if self.gradPrevOutput then
      self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
      nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
      gradOutput = self._gradOutputs[step]
   end
   
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local inputTable = {input, output, self.features}
   local gradInputTable = recurrentModule:updateGradInput(inputTable, gradOutputTable)
   local gradInput, gradFeatures
   gradInput, self.gradPrevOutput, gradFeatures= unpack(gradInputTable)
   self.gradFeaturesAcc:add(gradFeatures)
   if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   
   return gradInput
end

function GRUAttention:_accGradParameters(input, gradOutputTable, scale)
   assert(type(gradOutputTable) == 'table', 'gradOutput should be table')
   local gradOutput = gradOutputTable[1]

   local step = self.accGradParametersStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local inputTable = {input, output, self.features}
   gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]

   recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
   return gradInput
end
