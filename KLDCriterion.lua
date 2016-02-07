local KLDCriterion, _ = torch.class('nn.KLDCriterion', 'nn.Criterion')
function KLDCriterion:updateOutput(input, target)
    -- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local KLDelement = (input[2] + 1):add(-1,torch.pow(input[1],2)):add(-1,torch.exp(input[2]))
    self.output = 0.5 * torch.sum(KLDelement)
    return self.output
end

function KLDCriterion:updateGradInput(input, target)
	self.gradInput = {}
    self.gradInput[1] = (-input[1]):clone()
    self.gradInput[2] = (-torch.exp(input[2])):add(1):mul(0.5)

    return self.gradInput
end

local KLDCriterionn, _ = torch.class('nn.KLDCriterionn', 'nn.Criterion')

function KLDCriterionn:updateOutput(input, target)
    -- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local inp = input:chunk(2,2)
    local KLDelement = (inp[2] + 1):add(-1,torch.pow(inp[1],2)):add(-1,torch.exp(inp[2]))
    self.output = 0.5 * torch.sum(KLDelement)
    return self.output
end

function KLDCriterionn:updateGradInput(input, target)
	self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(input)
  local inp = input:chunk(2,2)
  local out = self.gradInput:chunk(2,2)
  out[1]:copy(-inp[1])
  out[2]:copy((-torch.exp(inp[2])):add(1):mul(0.5))
  return self.gradInput
end
