import torch
from torch import nn
import logging
import torch.nn.functional as f
from dgl import function as fn
from gnn.layer.hgatconv import NodeAttentionLayer
from gnn.layer.utils import LinearN
from typing import Callable, Union, Dict
import dgl


logger = logging.getLogger(__name__)


class GatedGATConv(nn.Module):
    """
    Gated GCN layer.
    It update bond, atom, and global features in sequence. See the BonDNet paper for
    details. This is a direct extension of the Residual Gated Graph ConvNets
    (https://arxiv.org/abs/1711.07553) by adding global features.
    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

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
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.
        Returns:
            updated node features.
        """
        g = g.to('cuda:1')
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

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
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

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        #两个节点直接联系的传递
        #------------------------------------
        def u2v_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["u2v"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
        def v2u_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["v2u"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
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
        v2u = g.edges["v2u"].data["m"]
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import torch.nn.functional as F
class GatedGCNConv(nn.Module):

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
        g = g.to('cuda:1')
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
        
        def message_funcAB(edges):
            z2 = torch.cat([edges.src['Ah'], edges.dst['Be']], dim=1)
            a = self.s(z2)
            return {'z' : edges.src['Ah'], 'e' : F.leaky_relu(a)}

        def reduce_funce(nodes):
            #print(nodes.mailbox['z'].size())
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            #print(alpha.size())
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            return {'e' : h}
        def message_funcBB(edges):
            z2 = torch.cat([edges.src['Be'], edges.dst['Be']], dim=1)
            a = self.s(z2)
            return {'z' : edges.src['Be'], 'e' : F.leaky_relu(a)}
        def message_funcCB(edges):
            z2 = torch.cat([edges.src['Cu'], edges.dst['Be']], dim=1)
            a = self.s(z2)
            return {'z' : edges.src['Cu'], 'e' : F.leaky_relu(a)}
        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (message_funcAB,reduce_funce),  # A * (h_i + h_j)
                "b2b": (message_funcBB,reduce_funce),  # B * e_ij
                "g2b": (message_funcCB,reduce_funce),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (message_funcAB,reduce_funce),  # A * (h_i + h_j)
                "1b2b": (message_funcBB,reduce_funce),  # B * e_ij
                "1g2b": (message_funcCB,reduce_funce),  # C * u
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
        def message_funcAA(edges):
            z2 = torch.cat([edges.src['Ah'], edges.dst['Ah']], dim=1)
            a = self.s(z2)
            return {'z' : edges.src['Ah'], 'e' : F.leaky_relu(a)}
        def reduce_funch(nodes):
            #print(nodes.mailbox['z'].size())
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            #print(alpha.size())
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            return {'h' : h}
        def message_funcCA(edges):
            z2 = torch.cat([edges.src['Cu'], edges.dst['Ah']], dim=1)
            a = self.s(z2)
            return {'z' : edges.src['Cu'], 'e' : F.leaky_relu(a)}
        g.multi_update_all(
            {
                "a2a": (message_funcAA,reduce_funch),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (message_funcCA,reduce_funch),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (message_funcAA,reduce_funch),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (message_funcCA,reduce_funch),  # F * u
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
    
class GatedGCNConv1(nn.Module):
    """
    Gated GCN layer.
    It update bond, atom, and global features in sequence. See the BonDNet paper for
    details. This is a direct extension of the Residual Gated Graph ConvNets
    (https://arxiv.org/abs/1711.07553) by adding global features.
    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

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
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.
        Returns:
            updated node features.
        """
        g = g.to('cuda:4')
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

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
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

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        #两个节点直接联系的传递
        #------------------------------------
        def u2v_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["u2v"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
        def v2u_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["v2u"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
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
        v2u = g.edges["v2u"].data["m"]
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
    
    
    
    
    
    
    
class GatedGCNConv2(nn.Module):
    """
    Gated GCN layer.
    It update bond, atom, and global features in sequence. See the BonDNet paper for
    details. This is a direct extension of the Residual Gated Graph ConvNets
    (https://arxiv.org/abs/1711.07553) by adding global features.
    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

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
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.
        Returns:
            updated node features.
        """
        g = g.to('cuda:4')
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

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        g.multi_update_all(
            {
                "1a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "1b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "1g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
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

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        
        
        g.multi_update_all(
            {
                "1a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "1b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "1g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        
        #两个节点直接联系的传递
        #------------------------------------
        def u2v_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["u2v"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
        def v2u_message_fn(edges):
            #print(edges.src["Dh"].size())
            c=1/(abs(edges.src["Dh"]-edges.dst["Dh"])+1)
            f.normalize(c, p=1, dim=1)
            #print(res.size())
            g.edges["v2u"].data.update({"m": self.w(torch.mul(c,edges.src["Dh"]))})
            return {"m": self.w(torch.mul(c,edges.src["Dh"]))}
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
        v2u = g.edges["v2u"].data["m"]
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
def select_not_equal(x, y):
    """Subselect an array from x, which is not equal to the corresponding element
    in y.
    Args:
        x (4D tensor): shape are d0: node batch dim, d1: number of edges dim,
            d2: selection dim, d3: feature dim
        y (2D tensor): shape are 0: nodes batch dim, 1: feature dim
    For example:
    >>> x =[[ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [0,1,2],
    ...         [6,7,8] ]
    ...     ],
    ...     [ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [3,4,5],
    ...         [6,7,8] ]
    ...     ]
    ...    ]
    >>>
    >>> y = [[0,1,2],
    ...      [3,4,5]]
    >>>
    >>> select_no_equal(x,y)
    ... [[[3,4,5],
    ...   [6,7,8]],
    ...  [[0,1,2],
    ...   [6,7,8]]
    Returns:
        3D tensor: of shape (d0, d1, d3)
    """
    d0, d1, d2, d3 = x.shape
    assert d2 == 2, f"Expect x.shape[2]==2, got {d2}"

    ## method 1, slow
    # rst = []
    # for x1, y1 in zip(x, y):
    #     xx = [x2[0] if not torch.equal(y1, x2[0]) else x2[1] for x2 in x1]
    #     rst.append(torch.stack(xx))
    # rst = torch.stack(rst)

    # method 2, a much faster version
    y = torch.repeat_interleave(y, d1 * d2, dim=0).view(x.shape)
    any_not_equal = torch.any(x != y, dim=3)
    # bool index
    idx1 = any_not_equal[:, :, 0].view(d0, d1, 1)
    idx2 = ~idx1
    idx = torch.cat([idx1, idx2], dim=-1)
    # select result
    rst = x[idx].view(d0, d1, -1)

    return rst