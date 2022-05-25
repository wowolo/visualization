import torch
from torch import nn

class NTK_Linear(nn.Module):

    def __init__(self, input_width, output_width, bias=True):
        
        super().__init__()

        self.A = nn.Parameter(torch.randn(input_width, output_width) / input_width**0.5)

        if bias:
            self.b = nn.Parameter(torch.randn(output_width))
        else:
            self.b = torch.zeros(output_width)
    


    def forward(self, x):
        return x @ self.A + self.b
        



class Stack_Core(nn.Module):

    def __init__(self, input_width, output_width, variable_width, skip_conn, linear_skip_conn, linear_skip_conn_width):
        
        super().__init__()

        self.skip_conn = skip_conn
        self.linear_skip_conn = linear_skip_conn
        self.linear_skip_conn_width = linear_skip_conn_width

        self.linear_1 = nn.Linear(input_width, variable_width)

        hidden_input_width = variable_width
        if self.skip_conn:
            hidden_input_width += input_width
        if self.linear_skip_conn:
            self.linear_skip = nn.Linear(input_width, self.linear_skip_conn_width)
            hidden_input_width += self.linear_skip_conn_width

        self.linear_2 = nn.Linear(hidden_input_width, output_width)

    

    def forward(self, x):

        hidden_input = nn.ReLU()(self.linear_1(x))

        if self.skip_conn:
            hidden_input = torch.cat((hidden_input, x), 1)

        if self.linear_skip_conn:
            linear_input = self.linear_skip(x)
            hidden_input = torch.cat((hidden_input, linear_input))

        y = self.linear_2(hidden_input)

        return y
