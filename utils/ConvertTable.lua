local ConvertTable, parent = torch.class('nn.ConvertTable', 'nn.JoinTable')

function ConvertTable:__init()
   parent.__init(self, 1)
end

-- output: T * nb_batch * nb_dim
function ConvertTable:updateOutput(input)
     local tensorSize = input[1]:size()
     local tableLength = #input
     local outputSize = torch.LongStorage():resize(1+tensorSize:size())
     outputSize[1] = tableLength
     for i=2, outputSize:size() do
           outputSize[i] = tensorSize[i-1]
     end

     -- check that every element in table has same size
     local flag = true
     for i=1, tableLength do
           local tempSize = input[i]:size()
             for j=1, tempSize:size() do
                   if tempSize[j] ~= tensorSize[j] then
                         flag = false
                         break
                     end
             end
     end

     -- call parent's funciton
     if flag then
           parent.updateOutput(self, input)
     else
             error()
             print('element has different size')
     end

     -- reshape
     self.output = self.output:view(outputSize)
   return self.output
end

function ConvertTable:updateGradInput(input, gradOutput)
     local equSize = torch.LongStorage():resize(gradOutput:dim()-1)
     equSize[1] = gradOutput:size(1) * gradOutput:size(2)
     for i=2, equSize:size() do
           equSize[i] = gradOutput:size(i+1)
     end

   parent.updateGradInput(self, input, gradOutput:view(equSize))
   return self.gradInput
end
