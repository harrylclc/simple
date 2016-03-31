local Linear3D, parent = torch.class('nn.Linear3D', 'nn.Linear')

function Linear3D:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
end

function Linear3D:updateOutput(input)
    local bs = input:size(1)
    local nSeq = input:size(2)
    local nDimIn = self.weight:size(2)
    local nDimOut = self.weight:size(1)

    self.output:resize(bs * nSeq, nDimOut)
    parent.updateOutput(self, input:view(-1, nDimIn))
    self.output = self.output:view(bs, nSeq, nDimOut)
    return self.output
end

function Linear3D:updateGradInput(input, gradOutput)
    local bs = input:size(1)
    local nSeq = input:size(2)
    local nDimIn = self.weight:size(2)
    local nDimOut = self.weight:size(1)

    self.gradInput:resize(bs * nSeq, nDimIn)
    parent.updateGradInput(self, input:view(-1, nDimIn), gradOutput:view(-1, nDimOut))
    self.gradInput = self.gradInput:view(bs, nSeq, nDimIn)
    return self.gradInput
end

function Linear3D:accGradParameters(input, gradOutput, scale)
    local nDimIn = self.weight:size(2)
    local nDimOut = self.weight:size(1)
    parent.accGradParameters(self, input:view(-1, nDimIn), gradOutput:view(-1, nDimOut), scale)
end
