from dgl import function as fn
from dgl.utils import expand_as_pair
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
from torch import nn
import logging
import torch.nn.functional as f
from dgl import function as fn
from gnn.layer.hgatconv import NodeAttentionLayer
from gnn.layer.utils import LinearN
from typing import Callable, Union, Dict
import dgl
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h
class MLP(nn.Module):
    """MLP with linear output"""
    #num_layers:共有多少层
    #input_dim：输入维度
    #hidden_dim：隐藏层维度，所有隐藏层维度都一样
    #hidden_dim：输出维度
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model这个时候只有一层MLP
        self.num_layers = num_layers
        self.output_dim = output_dim

        #层数合法性判断
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:#只有一层则按线性变换来玩，输入就是输出
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:#有多层则按下面代码处理
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))#第一层比较特殊，输入维度到隐藏层维度
            for layer in range(num_layers - 2):#中间隐藏层可以循环来玩，隐藏层维度到隐藏层维度
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))#最后一层，隐藏层维度到输出维度

            for layer in range(num_layers - 1):#除了最后一层都加BN
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):#前向传播
        if self.linear_or_not:#只有单层MLP
            # If linear model
            return self.linear(x)
        else:#多层MLP
            # If MLP
            h = x
            for i in range(self.num_layers - 1):#除最后一层外都加一个relu
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)#最后一层用线性变换把维度转到输出维度
class GatedGINConv(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 1,
        graph_norm: bool = False,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        
        
        
        #self.apply_func = ApplyNodeFunc()
        mlp = MLP(1, 800, 800, 800)
        self.apply_func = None
        self._aggregator_type = "sum"
        self.eps = torch.nn.Parameter(torch.FloatTensor([0]))
        
        
        
        
        
        # A, B, ... I are phi_1, phi_2, ..., phi_9 in the BonDNet paper
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        self.I = LinearN(input_dim, out_sizes, acts, use_bias)
        self.w = LinearN(800, out_sizes, acts, use_bias)
        self.z = LinearN(800, out_sizes, acts, use_bias)
        self.s = LinearN(2*800, out_sizes, acts, use_bias)
        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def reduce_fn_a2b(nodes):
        """
        Reduce `Eh_j` from atom nodes to bond nodes.
        Expand dim 1 such that every bond has two atoms connecting to it.
        This is to deal with the special case of single atom graph (e.g. H+).
        For such graph, an artificial bond is created and connected to the atom in
        `grapher`. Here, we expand it to let each bond connecting to two atoms.
        This is necessary because, otherwise, the reduce_fn wil not work since
        dimension mismatch.
        """
        x = nodes.mailbox["Eh_j"]
        if x.shape[1] == 1:
            x = x.repeat_interleave(2, dim=1)

        return {"Eh_j": x}

    @staticmethod
    def message_fn(edges):
        return {"Eh_j": edges.src["Eh_j"], "e": edges.src["e"]}

    @staticmethod
    def reduce_fn(nodes):
        Eh_i = nodes.data["Eh"]
        e = nodes.mailbox["e"]
        Eh_j = nodes.mailbox["Eh_j"]

        # TODO select_not_equal is time consuming; it might be improved by passing node
        #  index along with Eh_j and compare the node index to select the different one
        Eh_j = select_not_equal(Eh_j, Eh_i)
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

        return {"h": h}
    
    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        g = g.to('cuda:0')
        g = g.local_var()
        #print(feats)
        h = feats["atom"]
        e = feats["bond"]
        h2 = feats["atom2"]
        e2 = feats["bond2"]
        u = feats["global"]
        # for residual connection
        h_in = h
        e_in = e
        h_in2 = h2
        e_in2 = e2
        u_in = u
        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        g.nodes["atom2"].data.update({"Ah": self.A(h2), "Dh": self.D(h2), "Eh": self.E(h2)})
        g.nodes["bond2"].data.update({"Be": self.B(e2)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})
        
        _reducer = getattr(fn, self._aggregator_type)
        #aggregate_fn = fn.copy_u("h", "m")
        def add2(nodes):
            rst = (1 + self.eps) * nodes.data['Be'] + nodes.data["neigh"]
            #print("--------------------")
            #print(nodes.data['h'])
            #print("***************")
            #print( nodes.data['agg'])
            #print("--------------------")
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return {'e': rst}
        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"),_reducer("m", "neigh"),add2),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"),_reducer("m", "neigh"),add2),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"),_reducer("m", "neigh"),add2),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"),_reducer("m", "neigh"),add2),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"),_reducer("m", "neigh"),add2),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"),_reducer("m", "neigh"),add2),  # C * u
            },
            "sum",
        )

        e2 = g.nodes["bond2"].data["e"]
        if self.graph_norm:
            e2 = e2 * norm_bond
        if self.batch_norm:
            e2 = self.bn_node_e(e2)
        e2 = self.activation(e2)
        if self.residual:
            e2 = e_in2 + e2
        g.nodes["bond2"].data["e"] = e2
        
        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="1a2b")

        def add1(nodes):
            rst = (1 + self.eps) * nodes.data['Ah'] + nodes.data["neigh"]
            #print("--------------------")
            #print(nodes.data['h'])
            #print("***************")
            #print( nodes.data['agg'])
            #print("--------------------")
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return {'h': rst}
        
        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"),_reducer("m", "neigh"),add1),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"),_reducer("m", "neigh"),add1),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"),_reducer("m", "neigh"),add1),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"),_reducer("m", "neigh"),add1),  # F * u
            },
            "sum",
        )
        
        #两个节点直接联系的传递
        #------------------------------------
        def u2v_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["h"]-edges.dst["h"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["u2v"].data.update({"m": self.w(torch.mul(c,edges.src["h"]))})
            return {"m": self.w(torch.mul(c,edges.src["h"]))}
        def v2u_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["h"]-edges.dst["h"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["v2u"].data.update({"m": self.w(torch.mul(c,edges.src["h"]))})
            return {"m": self.w(torch.mul(c,edges.src["h"]))}
        def add(nodes):
            #print("--------------------")
            #print(nodes.data['h'])
            #print("***************")
            #print( nodes.data['agg'])
            #print("--------------------")
            return {'h': nodes.data['h']*0.8 + nodes.data['agg']*0.2}
        g.multi_update_all(
            {
                "u2v": (u2v_message_fn,fn.sum("m", "agg"),add),  # D * h_i
                "v2u": (v2u_message_fn,fn.sum("m", "agg"),add),
            },
            "sum",
        )
        u2v = g.edges["u2v"].data["m"]
        v2u = g.edges["v2u"].data["m"]#边的信息
        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h
        
        
        h2 = g.nodes["atom2"].data["h"]
        if self.graph_norm:
            h2 = h2 * norm_atom
        if self.batch_norm:
            h2 = self.bn_node_h(h2)
        h2 = self.activation(h2)
        if self.residual:
            h2 = h_in2 + h2
        g.nodes["atom2"].data["h"] = h2
        
        
        #--------------------------------------
        
        # update global feature u
        g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["bond"].data.update({"He": self.H(e)})
        g.nodes["atom2"].data.update({"Gh": self.G(h2)})
        g.nodes["bond2"].data.update({"He": self.H(e2)})
        g.nodes["global"].data.update({"Iu": self.I(u)})
        
        def add3(nodes):
            rst = (1 + self.eps) * nodes.data['Iu'] + nodes.data["neigh"]
            #print("--------------------")
            #print(nodes.data['h'])
            #print("***************")
            #print( nodes.data['agg'])
            #print("--------------------")
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return {'u': rst}
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "1a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "1b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )
        fn.copy_u("Dh", "m"),_reducer("m", "neigh"),add1
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"),_reducer("m", "neigh"),add3),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"),_reducer("m", "neigh"),add3),  # H * (mean_ij e_ij)
                "1a2g": (fn.copy_u("Gh", "m"),_reducer("m", "neigh"),add3),  # G * (mean_i h_i)
                "1b2g": (fn.copy_u("He", "m"),_reducer("m", "neigh"),add3),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"),_reducer("m", "neigh"),add3),  # I * u
            },
            "sum",
        )
        u = g.nodes["global"].data["u"]
        # do not apply batch norm if it there is only one graph
        if self.batch_norm and u.shape[0] > 1:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        h2 = self.dropout(h2)
        e2 = self.dropout(e2)
        u = self.dropout(u)

        feats = {"atom2": h2, "bond2": e2, "global": u,"atom": h, "bond": e,"u2v":u2v, "v2u":v2u}

        return feats