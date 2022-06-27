import torch
from torch import nn

class NTK_Linear(nn.Module):

    def __init__(self, input_width, output_width, bias=True, bias_tune=1):
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.bias = bias
        self.bias_tune = bias_tune

        self.initialize_parameters()



    def initialize_parameters(self):

        self.A = nn.Parameter(torch.randn(self.input_width, self.output_width))

        if self.bias:
            self.b = nn.Parameter(torch.randn(self.output_width))
            self.bias_tune = nn.Parameter(torch.ones(1) * self.bias_tune, requires_grad=False)
        else:
            self.b = torch.zeros(self.output_width)
    


    def forward(self, x):
        return x @ self.A / self.input_width**0.5 + self.b * self.bias_tune

    

    def reset_parameters(self):
        self.initialize_parameters()
        



class Stack_Core(nn.Module):

    def __init__(self, input_width, output_width, variable_width, skip_conn, linear_skip_conn, linear_skip_conn_width):
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.variable_width = variable_width

        self.skip_conn = skip_conn
        self.linear_skip_conn = linear_skip_conn
        self.linear_skip_conn_width = linear_skip_conn_width

        self.hidden_input_width = variable_width
        if self.skip_conn:
            self.hidden_input_width += input_width
        if self.linear_skip_conn:
            self.hidden_input_width += self.linear_skip_conn_width


        self.initialize_parameters()



    def initialize_parameters(self):

        self.linear_1 = nn.Linear(self.input_width, self.variable_width)
        
        if self.linear_skip_conn:
            self.linear_skip = nn.Linear(self.input_width, self.linear_skip_conn_width)

        self.linear_2 = nn.Linear(self.hidden_input_width, self.output_width)
    


    def forward(self, x):

        hidden_input = nn.ReLU()(self.linear_1(x))

        if self.skip_conn:
            hidden_input = torch.cat((hidden_input, x), 1)

        if self.linear_skip_conn:
            linear_input = self.linear_skip(x)
            hidden_input = torch.cat((hidden_input, linear_input), 1)

        y = self.linear_2(hidden_input)

        return y

    

    def reset_parameters(self):
        self.initialize_parameters()



# class abc_Layer(nn.Module):

#     def __init__(self, input_width, output_width, a, b, c):
        
#         super().__init__()

#         self.input_width = input_width
#         self.output_width = output_width
#         self.variable_width = variable_width

#         self.skip_conn = skip_conn
#         self.linear_skip_conn = linear_skip_conn
#         self.linear_skip_conn_width = linear_skip_conn_width

#         self.hidden_input_width = variable_width
#         if self.skip_conn:
#             self.hidden_input_width += input_width
#         if self.linear_skip_conn:
#             self.hidden_input_width += self.linear_skip_conn_width


#         self.initialize_parameters()



#     def initialize_parameters(self):

#         self.linear_1 = nn.Linear(self.input_width, self.variable_width)
        
#         if self.linear_skip_conn:
#             self.linear_skip = nn.Linear(self.input_width, self.linear_skip_conn_width)

#         self.linear_2 = nn.Linear(self.hidden_input_width, self.output_width)
    


#     def forward(self, x):

#         hidden_input = nn.ReLU()(self.linear_1(x))

#         if self.skip_conn:
#             hidden_input = torch.cat((hidden_input, x), 1)

#         if self.linear_skip_conn:
#             linear_input = self.linear_skip(x)
#             hidden_input = torch.cat((hidden_input, linear_input), 1)

#         y = self.linear_2(hidden_input)

#         return y

    

#     def reset_parameters(self):
#         self.initialize_parameters()