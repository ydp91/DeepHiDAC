import torch
from torch import nn
from torch.autograd import Variable

def activation_layer(act_name, negative_slope=0.1):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU()
        elif act_name.lower() == 'leaky_relu':
            act_layer = nn.LeakyReLU(negative_slope)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class MLP(nn.Module):
    """docstring for MLP"""

    def __init__(
            self,
            input_size,
            layer_sizes,
            activation=nn.ReLU(),
            dropout=0.1,
            output_act=nn.Identity()):
        super(MLP, self).__init__()
        self.dropout = dropout
        layer_sizes = [input_size] + layer_sizes
        self.mlp = self.build_mlp(layer_sizes, output_act, activation)

    def build_mlp(self, layer_sizes, output_act=nn.Identity(), activation=nn.ReLU()):
        layers = []
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else output_act
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act]
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_units, activation, dropout=0):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.lin = MLP(in_dim, hidden_units, activation, dropout, activation)


    def forward(self, x):
        return self.lin(x) + x


class ResDNN(nn.Module):
    """The Multi Layer Percetron with Residuals
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most
          common situation would be a 2D input with shape
          ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of list, which contains the layer number and
          units in each layer.
            - e.g., [[5], [5,5], [5,5]]
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied
          to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation.
    """

    def __init__(self, input_dim, hidden_units, activation=nn.ReLU(), dropout=0, use_bn=False):
        super(ResDNN, self).__init__()
        if input_dim != hidden_units[0][0]:
            raise ValueError('In ResBlock, the feature size must be equal to the hidden \
                size! input_dim:{}, hidden_size: {}'.format(input_dim, hidden_units[0]))
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        self.hidden_units = hidden_units
        self.hidden_units[0] = [input_dim] + self.hidden_units[0]
        self.resnet = nn.ModuleList(
            [ResBlock(h[0], h[1:], activation, use_bn) for h in hidden_units])

    def forward(self, x):
        for i in range(len(self.hidden_units)):
            out = self.resnet[i](x)
            out = self.dropout(out)
        return out








class PositionalEncoding(nn.Module):

    def __init__(self, e_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, e_dim).float()
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = 10000.0 ** (torch.arange(0., e_dim, 2.) / e_dim)

        # 偶数位计算sin, 奇数位计算cos
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False).cuda()
        return self.dropout(x)













class NN_D(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dropout):
        super(NN_D, self).__init__()

        self.pos_to_dim = nn.Sequential(
            nn.Linear(6, dim),
            nn.LeakyReLU()
        )
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # 时间序列特征提取

        self.pos_encoder = PositionalEncoding(dim)

        self.dim_to_out = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, rel_traj,speed,target):
        # 聚合时间特征
        traj_cat = torch.concat([rel_traj, speed,target], dim=-1)
        pos_emb = self.pos_to_dim(traj_cat)  # n,len,dim
        t_in = self.pos_encoder(pos_emb)
        mask = nn.Transformer.generate_square_subsequent_mask(rel_traj.size(1)).to(rel_traj)
        t_out = self.time_encoder(t_in, mask=mask) #时间特征输出
        out = self.dim_to_out(t_out) #每一步预测的为该步的下个相对位置，所以没取最后一个位置（无法验证损失）
        return out

class NN_C(nn.Module):
    def __init__(self, dim, dropout):
        super(NN_C, self).__init__()

        self.embed = ResBlock(4,[dim,4],activation_layer('leaky_relu'),dropout)
        self.out=nn.Sequential(nn.Linear(4, 2),nn.Sigmoid())

    def forward(self, traj, speed, seq_start_end):
        indexs, vectors = [], []
        for i, se in enumerate(seq_start_end):
            index, vector = self.find_neighbors(traj[se[0]:se[1],:], speed[se[0]:se[1],:])
            indexs.append(index)
            expand_speed = speed[se[0]:se[1]].unsqueeze(1).expand(se[1] - se[0],se[1] - se[0], traj.size(1), 2)
            vector = torch.concat([vector, expand_speed], dim=-1)
            embed=self.embed(vector)
            out=self.out(embed)
            vectors.append(out)
        return indexs, vectors  # index需考虑的索引,agent i(当前agent),agent j(其他agent),pos_num, vectors 全部相对位置向量计算（基于速度计算）

    def find_neighbors(self,people_positions, people_velocities,sensing=2,angle_threshold=0.8333*3.1415926):
        n, len, _ = people_positions.shape

        # 将人的位置进行广播
        position_data_expanded = people_positions.unsqueeze(1).expand(n, n, len, 2)
        vector = people_positions.unsqueeze(0) - position_data_expanded
        distances = torch.norm(vector, dim=3)
        directions = vector / (distances.unsqueeze(3) + 1e-5)

        dot_products = (people_velocities.unsqueeze(1) * directions).sum(dim=3)
        speeds = torch.norm(people_velocities.unsqueeze(1), dim=-1) + 1e-5
        cosine_similarity = dot_products / speeds
        angles = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))

        # 创建一个布尔掩码，指示距离小于2米的情况
        mask = (distances <= sensing) & (angles <= angle_threshold / 2)

        # 返回距离小于2米的障碍物的索引
        index = mask.nonzero(as_tuple=False)

        return index, vector  # 索引和全部差值

class NN_O(nn.Module):
    def __init__(self, dim, dropout):
        super(NN_O, self).__init__()

        self.embed = ResBlock(4, [dim, 4], activation_layer('leaky_relu'), dropout)
        self.out = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())

    def forward(self, traj, speed,obstacle, seq_start_end):
        indexs,vectors = [],[]
        for i, se in enumerate(seq_start_end):
            index,vector=self.find_obstacles(traj[se[0]:se[1]],speed[se[0]:se[1]],obstacle[i])
            indexs.append(index)
            expand_speed=speed[se[0]:se[1]].unsqueeze(2).expand(se[1]-se[0], traj.size(1), obstacle[i].size(0), 2)
            vector=torch.concat([vector,expand_speed],dim=-1)
            embed = self.embed(vector)
            out=self.out(embed)
            vectors.append(out)
        return indexs,vectors #index需考虑的障碍索引 人ID,位置,障碍物, vectors 全部相对位置向量计算（基于速度计算）


    def find_obstacles(self,people_positions, people_velocities, obstacle_positions,sensing=2,angle_threshold=0.8333*3.1415926):
        n, len, _ = people_positions.shape
        num_obstacles, _ = obstacle_positions.shape

        # 将人的位置和障碍物位置扩展为相同形状以进行广播
        people_positions_expanded = people_positions.unsqueeze(2).expand(n, len, num_obstacles, 2)
        # obstacle_positions_expanded = obstacle_positions.unsqueeze(0).unsqueeze(0).expand(n, len, num_obstacles, 2)

        # 计算人和障碍物之间的距离

        vector = obstacle_positions-people_positions_expanded
        distances = torch.norm(vector, dim=3)
        directions = vector / (distances.unsqueeze(3) + 1e-5)

        dot_products = (people_velocities.unsqueeze(2) * directions).sum(dim=3)
        speeds = torch.norm(people_velocities, dim=2).unsqueeze(2) + 1e-5
        cosine_similarity = dot_products / speeds
        angles = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))

        # 创建一个布尔掩码，指示距离小于2米的情况
        mask = (distances <= sensing) & (angles <= angle_threshold / 2)

        # 返回距离小于2米的障碍物的索引
        index = mask.nonzero(as_tuple=False)

        return index,vector #索引和全部差值



class Encoder(nn.Module):
    def __init__(self, dim,mlp_dim,heads, dropout,depth):
        super(Encoder, self).__init__()
        self.tinfo_embed = nn.Sequential(
            nn.Linear(8, dim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # 时间序列特征提取
        self.fc_mu = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(dim*2, dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(dim*2, dim)
        )

    def forward(self, x, traj,speed,pred):
        x = torch.cat((x, traj,speed,pred), dim=-1)  # Concatenate the condition data
        x = self.tinfo_embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x)
        x = self.transformer(x, mask=mask)  # 时间特征输出
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dropout, depth):
        super(Decoder, self).__init__()
        self.tinfo_embed = nn.Sequential(
            nn.Linear(6, dim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # 时间序列特征提取


        self.fc = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(dim, 2),
            nn.Tanh()
        )

    def forward(self, z, traj,speed,pred):
        condition = torch.cat(( traj, speed,pred), dim=-1)
        cond = self.tinfo_embed(condition)
        mask = nn.Transformer.generate_square_subsequent_mask(condition.size(1)).to(condition)
        cond = self.transformer(cond, mask=mask)
        l_in=torch.concat([cond,z],dim=-1)
        x_hat = self.fc(l_in)
        return x_hat



class CVAE(nn.Module):
    def __init__(self, dim,mlp_dim,heads, dropout,depth):
        super(CVAE, self).__init__()
        self.encoder = Encoder(dim,mlp_dim,heads, dropout,depth)
        self.decoder = Decoder(dim,mlp_dim,heads, dropout,depth)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, traj,speed,pred):
        mu, logvar = self.encoder(x,  traj,speed,pred)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z,  traj,speed,pred)
        return x_hat, mu, logvar